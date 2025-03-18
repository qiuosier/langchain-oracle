# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from langchain_oci.chat_models.oci_data_science import ChatOCIModelDeployment
from langchain_oci.chat_models.oci_generative_ai import ChatOCIGenAI

__all__ = ["ChatOCIGenAI", "ChatOCIModelDeployment"]
