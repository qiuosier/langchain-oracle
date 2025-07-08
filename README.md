# 🦜️🔗 LangChain 🤝 Oracle

Welcome to the official repository for LangChain integration with [Oracle Cloud Infrastructure (OCI)](https://cloud.oracle.com/). This library provides native LangChain components for interacting with Oracle's AI services—combining support for **OCI Generative AI** and **OCI Data Science**.

## Features

- **LLMs**: Includes LLM classes for OCI services like [Generative AI](https://cloud.oracle.com/ai-services/generative-ai) and [ModelDeployment Endpoints](https://cloud.oracle.com/ai-services/model-deployment), allowing you to leverage their language models within LangChain.
- **Agents**: Includes Runnables to support [Oracle Generative AI Agents](https://www.oracle.com/artificial-intelligence/generative-ai/agents/), allowing you to leverage Generative AI Agents within LangChain and LangGraph.
- **More to come**: This repository will continue to expand and offer additional components for various OCI services as development progresses.

> This project merges and replaces earlier OCI integrations from the `langchain-community` repository and unifies contributions from Oracle's GenAI and Data Science teams.
> All integrations in this package assume that you have the credentials setup to connect with oci services.

---

## Installation


```bash
pip install -U langchain-oci
```

---

## Quick Start

This repository includes two main integration categories:

- [OCI Generative AI](#oci-generative-ai-examples)
- [OCI Data Science (Model Deployment)](#oci-data-science-model-deployment-examples)


---

## OCI Generative AI Examples

### 1. Use a Chat Model

`ChatOCIGenAI` class exposes chat models from OCI Generative AI.

```python
from langchain_oci import ChatOCIGenAI

llm = ChatOCIGenAI()
llm.invoke("Sing a ballad of LangChain.")
```

### 2. Use a Completion Model
`OCIGenAI` class exposes LLMs from OCI Generative AI.

```python
from langchain_oci import OCIGenAI

llm = OCIGenAI()
llm.invoke("The meaning of life is")
```

### 3. Use an Embedding Model
`OCIGenAIEmbeddings` class exposes embeddings from OCI Generative AI.

```python
from langchain_oci import OCIGenAIEmbeddings

embeddings = OCIGenAIEmbeddings()
embeddings.embed_query("What is the meaning of life?")
```


## OCI Data Science Model Deployment Examples

### 1. Use a Chat Model

You may instantiate the OCI Data Science model with the generic `ChatOCIModelDeployment` or framework specific class like `ChatOCIModelDeploymentVLLM`.

```python
from langchain_oci.chat_models import ChatOCIModelDeployment, ChatOCIModelDeploymentVLLM

# Create an instance of OCI Model Deployment Endpoint
# Replace the endpoint uri with your own
endpoint = "https://modeldeployment.<region>.oci.customer-oci.com/<ocid>/predict"

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]

chat = ChatOCIModelDeployment(
    endpoint=endpoint,
    streaming=True,
    max_retries=1,
    model_kwargs={
        "temperature": 0.2,
        "max_tokens": 512,
    },  # other model params...
    default_headers={
        "route": "/v1/chat/completions",
        # other request headers ...
    },
)
chat.invoke(messages)

chat_vllm = ChatOCIModelDeploymentVLLM(endpoint=endpoint)
chat_vllm.invoke(messages)
```

### 2. Use a Completion Model
You may instantiate the OCI Data Science model with `OCIModelDeploymentLLM` or `OCIModelDeploymentVLLM`.

```python
from langchain_oci.llms import OCIModelDeploymentLLM, OCIModelDeploymentVLLM

# Create an instance of OCI Model Deployment Endpoint
# Replace the endpoint uri and model name with your own
endpoint = "https://modeldeployment.<region>.oci.customer-oci.com/<ocid>/predict"

llm = OCIModelDeploymentLLM(
    endpoint=endpoint,
    model="odsc-llm",
)
llm.invoke("Who is the first president of United States?")

vllm = OCIModelDeploymentVLLM(
    endpoint=endpoint,
)
vllm.invoke("Who is the first president of United States?")
```

### 3. Use an Embedding Model
You may instantiate the OCI Data Science model with the `OCIModelDeploymentEndpointEmbeddings`.

```python
from langchain_oci.embeddings import OCIModelDeploymentEndpointEmbeddings

# Create an instance of OCI Model Deployment Endpoint
# Replace the endpoint uri with your own
endpoint = "https://modeldeployment.<region>.oci.customer-oci.com/<ocid>/predict"

embeddings = OCIModelDeploymentEndpointEmbeddings(
    endpoint=endpoint,
)

query = "Hello World!"
embeddings.embed_query(query)

documents = ["This is a sample document", "and here is another one"]
embeddings.embed_documents(documents)
```


## Contributing

This project welcomes contributions from the community. Before submitting a pull request, please [review our contribution guide](./CONTRIBUTING.md)

## Security

Please consult the [security guide](./SECURITY.md) for our responsible security vulnerability disclosure process

## License

Copyright (c) 2025 Oracle and/or its affiliates.

Released under the Universal Permissive License v1.0 as shown at
<https://oss.oracle.com/licenses/upl/>
