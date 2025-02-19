# langchain-oci

This package contains the LangChain integrations with oci.

## Installation

```bash
pip install -U langchain-oci
```
All integrations in this package assume that you have the credentials setup to connect with oci services.

## Chat Models

`ChatOCIGenAI` class exposes chat models from OCI Generative AI.

```python
from langchain_oci import ChatOCIGenAI

llm = ChatOCIGenAI()
llm.invoke("Sing a ballad of LangChain.")
```

## Embeddings

`OCIGenAIEmbeddings` class exposes embeddings from OCI Generative AI.

```python
from langchain_oci import OCIGenAIEmbeddings

embeddings = OCIGenAIEmbeddings()
embeddings.embed_query("What is the meaning of life?")
```

## LLMs
`OCIGenAI` class exposes LLMs from OCI Generative AI.

```python
from langchain_oci import OCIGenAI

llm = OCIGenAI()
llm.invoke("The meaning of life is")
```