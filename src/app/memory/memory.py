from __future__ import annotations

from typing import Dict, Optional, List

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda,RunnablePassthrough

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_milvus import Milvus

class MemoryManager:
    """
    长短期记忆管理器：
    - 短期：InMemoryChatMessageHistory（按 session_id 保存对话消息）
    - 长期：Milvus VectorStore（按 user_id 写入/检索语义记忆）
    """
    def __init__(
        self,
        *,
        embeddings: Embeddings,
        milvus_uri: str = "http://localhost:19530",
        collection_name: str = "memories",
        drop_old: bool = True,
        k: int = 6,
    ):
        self._chat_store: Dict[str, InMemoryChatMessageHistory] = {}
        self.k = k
        self.vs = Milvus(
            embedding_function=embeddings,
            connection_args={"uri": milvus_uri},
            collection_name=collection_name,
            # 你可以沿用 IVF_FLAT + COSINE（你之前也是这个思路）
            index_params={"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 128}},
            drop_old=drop_old,  # 开发期 True；上线一般 False
        )
        self.retriever = self.vs.as_retriever(search_kwargs={"k": k})

    # --------------------
    # 短期记忆（对话历史）
    # --------------------
    def get_session_history(self, session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in self._chat_store:
            self._chat_store[session_id] = InMemoryChatMessageHistory()
        return self._chat_store[session_id]
    
    # --------------------
    # 长期记忆（语义记忆）
    # --------------------

    def remember(self,*,text:str,user_id:str,kind:str="fact",source:str="caht")->List[str]:
        """写入一条长期记忆。"""
        doc = Document(page_content=text, metadata={"user_id": user_id, "kind": kind, "source": source})
        # add_documents 返回 ids（不同版本返回形式可能略有差异，但总体可用）
        return self.vs.add_documents([doc])
    
    def recall(self,*,query:str,user_id:str)->List[Document]:
        """
        检索长期记忆。
        Milvus retriever 支持在 invoke 时传 filter（官方示例：retriever.invoke(..., filter={...})）。
        """
        return self.retriever.invoke(query, filter={"user_id": user_id})
    
    # --------------------
    # 构建“带长短期记忆”的聊天 runnable
    # --------------------
    def build_chat(self, llm):
        """
        返回一个 runnable：输入 {"input": str}，并通过 configurable.session_id 维护短期对话历史；
        同时需要通过 configurable.user_id 来做长期记忆检索隔离。
        """
        def _format_docs(docs: List[Document]) -> str:
            if not docs:
                return ""
            return "\n".join([f"- {d.page_content}" for d in docs])

        # 从输入里拿 query，再用 user_id 去检索长期记忆，最后格式化成 context
        def _get_context(payload: dict) -> str:
            query = payload["input"]
            user_id = payload.get("user_id") or "default"
            docs = self.recall(query=query, user_id=user_id)
            return _format_docs(docs)

        context_runnable = RunnableLambda(_get_context)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "你是一个有记忆的助手。\n长期记忆（可能相关）：\n{context}"),
                MessagesPlaceholder("history"),
                ("human", "{input}"),
            ]
        )

        chain = (
            {"input": RunnablePassthrough(), "context": context_runnable}
            | prompt
            | llm
            | StrOutputParser()
        )

        # RunnableWithMessageHistory：官方定义就是“包一层 runnable 并负责读写历史”。:contentReference[oaicite:3]{index=3}
        with_history = RunnableWithMessageHistory(
            chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )
        return with_history