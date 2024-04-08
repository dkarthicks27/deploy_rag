import os
import time
from enum import Enum
from typing import List

import streamlit as st
from llama_index.core import Document
from llama_index.core import Settings, QueryBundle
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import VectorStoreIndex, SimpleKeywordTableIndex
from llama_index.core import get_response_synthesizer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SentenceTransformerRerank, SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import (
    BaseRetriever,
    KeywordTableSimpleRetriever, QueryFusionRetriever,
)
from llama_index.core.postprocessor import KeywordNodePostprocessor
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.retrievers.bm25 import BM25Retriever
from llama_parse import LlamaParse
from llmsherpa.readers import LayoutPDFReader
from stqdm import stqdm
from tqdm import tqdm

# loading the page config settings
st.set_page_config('Query your pdf', layout='centered')
if 'index' not in st.session_state:
    st.session_state.index = None

if 'curr_llm' not in st.session_state:
    st.session_state.curr_llm = None

if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = None

if 'document' not in st.session_state:
    st.session_state.document = None

if 'queryEngine' not in st.session_state:
    st.session_state.queryEngine = None

if 'reranker' not in st.session_state:
    st.session_state.reranker = []

if 'retriever' not in st.session_state:
    st.session_state.retriever = None

if 'nodeParser' not in st.session_state:
    st.session_state.nodeParser = SentenceSplitter.from_defaults()

if 'rag' not in st.session_state:
    st.session_state.rag = None

if 'LLAMAPARSE_API_KEY' not in st.session_state:
    st.session_state.LLAMAPARSE_API_KEY = None


class Parser(str, Enum):
    LLMSherpa = 'LLMSherpa'
    LlamaParse = 'LlamaParse'


@st.cache_resource
def set_reranker():
    reranker = SentenceTransformerRerank(top_n=4, model="BAAI/bge-reranker-base")
    st.session_state.reranker.append(reranker)


class HybridRetriever(BaseRetriever):
    def __init__(self, vector_retriever, bm25_retriever):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        super().__init__()

    def _retrieve(self, query, **kwargs):
        bm25_nodes = self.bm25_retriever.retrieve(query, **kwargs)
        vector_nodes = self.vector_retriever.retrieve(query, **kwargs)

        # combine the two lists of nodes
        all_nodes = []
        node_ids = set()
        for n in bm25_nodes + vector_nodes:
            if n.node.node_id not in node_ids:
                all_nodes.append(n)
                node_ids.add(n.node.node_id)
        return all_nodes


class CustomRetriever(BaseRetriever):
    def __init__(self, vec_ret, key_ret, mode="AND"):
        super().__init__()
        self.vec_ret = vec_ret
        self.key_ret = key_ret
        if mode not in ('AND', 'OR'):
            raise ValueError("Invalid mode")
        self._mode = mode

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        vector_nodes = self.vec_ret.retrieve(query_bundle)
        keyword_nodes = self.key_ret.retrieve(query_bundle)

        combined_dict = {}
        for n in vector_nodes + keyword_nodes:
            if n.node.node_id not in combined_dict:
                combined_dict[n.node.node_id] = n

        vector_ids = set(n.node.node_id for n in vector_nodes)
        keyword_ids = set(n.node.node_id for n in keyword_nodes)

        if self._mode == 'AND':
            # we will be doing an intersection
            ids = vector_ids.intersection(keyword_ids)
        else:
            ids = vector_ids.union(keyword_ids)
        retrieved_nodes = [combined_dict[rid] for rid in ids]

        return retrieved_nodes


@st.cache_data(show_spinner=False)
def is_open_ai_key_valid(openai_api_key, model: str) -> bool:
    if not openai_api_key:
        st.error("Please enter your OpenAI API key in the sidebar!")
        return False
    try:
        client = OpenAI(api_key=openai_api_key, model=model)
        client.complete("test")

    except Exception as e:
        st.error(f"{e.__class__.__name__}: {e}")
        return False

    return True


@st.cache_data(experimental_allow_widgets=True)
def load_llm_settings(model="gpt-3.5-turbo", temperature=0.7):
    with st.expander("Openai api key"):
        with st.form("OpenAI key", border=False):
            key = st.text_input("Enter OpenAI API key", placeholder="Enter OpenAI API Key", label_visibility="collapsed")

            submit = st.form_submit_button("Submit")
            if submit:
                if not is_open_ai_key_valid(model=model, openai_api_key=key):
                    st.stop()
                else:
                    st.success("OpenAI key set successfully", icon="✅")
                    Settings.llm = OpenAI(api_key=key, model=model, temperature=temperature)
                    st.session_state.rag = True


@st.cache_data
def load_embedding_settings(embedding_model="sentence-transformers/all-mpnet-base-v2"):
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=embedding_model
    )


@st.cache_data
def loader_(chunk_size=256, chunk_overlap=25):
    pass


@st.cache_data
def save_pdf_from_bytes(pdf_bytes, filename):
    with open(filename, 'wb') as f:
        f.write(pdf_bytes)
    full_path = os.path.abspath(filename)
    success = st.success('File uploaded successfully!', icon="✅")
    time.sleep(0.5)
    success.empty()
    return full_path


@st.cache_resource
def generate_embedding(pdf_url, opt, api_key=None):
    if opt == Parser.LLMSherpa:
        gen_using_llmsherpa(pdf_url)
    else:
        print(f"api_key : {api_key}")
        if api_key is None:
            api_key = st.session_state.LLAMAPARSE_API_KEY
        gen_using_llamaParse(pdf_url, api_key)


@st.cache_resource
def gen_using_llamaParse(pdf_url, api_key):
    try:
        with st.status("Generating Embeddings...") as s:
            st.write("Loading the document and chunking...")
            print(f"api key is : {api_key}")
            parser = LlamaParse(
                api_key=api_key,
                result_type="markdown"
            )
            documents = parser.load_data(pdf_url)
            st.session_state.document = documents
            st.write("Generating vectors over documents...")
            index = VectorStoreIndex.from_documents(documents)
            if Settings.embed_model != 'text-embedding-ada-002':
                st.write("Persisting the index...")
                index.storage_context.persist(persist_dir=f"{pdf_url}_llamaparse_index")
            else:
                print("\n\nEmbedding model is: ", Settings.embed_model, end="\n\n\n")
    except Exception as e:
        st.error(e)
        print(f"\n\n\nThe error is: {e}\n\n\n")


@st.cache_resource
def gen_using_llmsherpa(pdf_url):
    with st.status("Generating Embeddings...") as s:
        st.write("Loading the document and chunking...")
        llmsherpa_api_url = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
        pdf_reader = LayoutPDFReader(llmsherpa_api_url)
        doc = pdf_reader.read_pdf(pdf_url)
        final_doc = []
        for chunk in tqdm(doc.chunks(), desc="converting chunk..."):
            final_doc.append(Document(text=chunk.to_context_text(), extra_info={}))
        st.session_state.document = final_doc
        st.write("Generating vectors over chunks...")
        index = VectorStoreIndex([])

        if Settings.embed_model != 'text-embedding-ada-002':
            for chunk in stqdm(doc.chunks(), desc="converting chunk..."):
                index.insert(Document(text=chunk.to_context_text(), extra_info={}))

            st.write("Persisting the index...")
            index.storage_context.persist(persist_dir=f"{pdf_url}_llmsherpa_index")
        else:
            print("\n\nEmbedding model is: ", Settings.embed_model, end="\n\n\n")

    st.session_state.index = index
    print(st.session_state.index, "ran successfully")
    s.empty()


st.title("ASK PDF")
with st.sidebar:
    pdf_file = st.file_uploader(label=":blue[**SUBMIT BELOW**]", type=['pdf'], label_visibility='visible')
    path = None
    if pdf_file is not None and pdf_file.type == 'application/pdf':
        opt = st.selectbox("Preferred Parser", options=[Parser.LLMSherpa.value, Parser.LlamaParse.value], index=0,
                           placeholder="Select a parser")
        api_key = None
        if opt == Parser.LlamaParse:
            with st.expander("LLamaParse API key"):
                with st.form("LLamaParse API key", border=False):
                    key = st.text_input("Enter LLamaParse API key", placeholder="Enter LLamaParse API key",
                                        label_visibility="collapsed")
                    submit_llama = st.form_submit_button("Submit")

                    if submit_llama:
                        st.success("LlamaParse Key API entered", icon="✅")
                        if st.session_state.LLAMAPARSE_API_KEY is None:
                            st.session_state.LLAMAPARSE_API_KEY = key
                        print(f"is api key getting set: {api_key}")

        path = save_pdf_from_bytes(pdf_file.getvalue(), pdf_file.name)

    if path is not None:
        load_llm_settings()
        load_embedding_settings()
        print(f"The first api key is : {api_key}")
        st.button('Generate Embedding', type='secondary', key='gen_kd', on_click=generate_embedding,
                  args=[pdf_file.name, opt, api_key])


@st.cache_data
def update_similaritypostprocessor(cutoff):
    postprocessor = SimilarityPostprocessor(similarity_cutoff=cutoff)
    if postprocessor not in st.session_state.reranker:
        st.session_state.reranker.append(postprocessor)


@st.cache_data
def update_keywordNodePostprocessor(allowed_list, excluded_list):
    postprocessor = KeywordNodePostprocessor(required_keywords=allowed_list, exclude_keywords=excluded_list)
    if postprocessor not in st.session_state.reranker:
        st.session_state.reranker.append(postprocessor)


class FUSION_MODES(str, Enum):
    """Enum for different fusion modes."""

    RECIPROCAL_RANK = "reciprocal_rerank"  # apply reciprocal rank fusion
    RELATIVE_SCORE = "relative_score"  # apply relative score fusion
    DIST_BASED_SCORE = "dist_based_score"  # apply distance-based score fusion
    SIMPLE = "simple"  # simple re-ordering of results based on original scores


@st.cache_data
def set_retrieval_settings(retriever, use_rrf):
    if retriever == 'Hybrid':
        vector_retriever = st.session_state.index.as_retriever(similarity_top_k=5)
        bm25_retriever = BM25Retriever.from_defaults(docstore=st.session_state.index.docstore, similarity_top_k=5)
        if use_rrf:
            st.session_state.retriever = QueryFusionRetriever([vector_retriever, bm25_retriever],
                                                              similarity_top_k=5,
                                                              num_queries=1,
                                                              mode=FUSION_MODES.RECIPROCAL_RANK,
                                                              use_async=True,
                                                              verbose=True,
                                                              )
        else:
            st.session_state.retriever = QueryFusionRetriever([vector_retriever, bm25_retriever],
                                                              similarity_top_k=5,
                                                              num_queries=1,
                                                              mode=FUSION_MODES.SIMPLE,
                                                              use_async=True,
                                                              verbose=True,
                                                              )

    elif retriever == 'Vector + Keyword':
        vector_retriever = st.session_state.index.as_retriever(similarity_top_k=5)
        keyword_index = SimpleKeywordTableIndex.from_documents(st.session_state.document)
        keyword_retriever = KeywordTableSimpleRetriever(index=keyword_index)
        st.session_state.retriever = CustomRetriever(vector_retriever, keyword_retriever, mode='OR')

    elif retriever == 'Vector':
        st.session_state.retriever = st.session_state.index.as_retriever(similarity_top_k=5)

    elif retriever == 'BM25':
        st.session_state.retriever = BM25Retriever.from_defaults(docstore=st.session_state.index.docstore,
                                                                 similarity_top_k=5)

    elif retriever == 'Keyword':
        keyword_index = SimpleKeywordTableIndex.from_documents(st.session_state.document)
        st.session_state.retriever = KeywordTableSimpleRetriever(index=keyword_index)


def load_llmsherpa_doc_from_local(file_name, api_key=None):
    try:
        llmsherpa_api_url = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
        pdf_reader = LayoutPDFReader(llmsherpa_api_url)
        doc = pdf_reader.read_pdf(file_name)
        final_doc = []
        for chunk in tqdm(doc.chunks(), desc="converting chunk..."):
            final_doc.append(Document(text=chunk.to_context_text(), extra_info={}))
        st.session_state.document = final_doc
    except Exception as e:
        st.error(f"Error with LLMsherpa: {e}\n Please try Llama Parse Instead.")
        print(e)


def load_llamaparse_doc_from_local(file_name, api_key=None):
    parser = LlamaParse(
        api_key=api_key,
        result_type="markdown"
    )
    documents = parser.load_data(file_name)
    st.session_state.document = documents


@st.cache_resource
def load_index_from_local(file_name, option, api_key):
    try:
        with st.status("Generating Embeddings...") as s:
            st.write("Loading the document and chunking...")
            if option == Parser.LLMSherpa:
                load_llmsherpa_doc_from_local(file_name, api_key)
                persist_dir = f"{file_name}_llmsherpa_index"

            elif option == Parser.LlamaParse:
                if not api_key:
                    api_key = st.session_state.LLAMAPARSE_API_KEY
                load_llamaparse_doc_from_local(file_name, api_key)
                persist_dir = f"{file_name}_llamaparse_index"

            if os.path.exists(persist_dir):
                storage_context = StorageContext.from_defaults(persist_dir=f"{file_name}_llmsherpa_index")
            else:
                s.update(label="Index not found!! Generate Embeddings", state="error", expanded=True)
                return
            st.write('Searching for index for the selected document..')
            index = load_index_from_storage(storage_context)
            st.session_state.index = index
        s.update(label="Index Loaded Successfully", state="complete", expanded=False)
        s.empty()
    except Exception as e:
        st.exception(e)


if path is not None:
    with st.sidebar as sk:
        st.header("**RAG Settings**")
        index_usage = True
        if path is not None:
            index_usage = False

        filter_ = st.toggle("Use Existing Index for this file", disabled=index_usage)
        if filter_:
            load_index_from_local(pdf_file.name, opt, api_key)

        st.write("")
        st.write("")
        if st.session_state.index and st.session_state.rag:
            with st.expander("LLM"):
                with st.form("LLM settings", clear_on_submit=True, border=False):
                    llm_current = st.selectbox("LLM provider", placeholder="Select a LLM", options=('gpt-3.5-turbo',
                                                                                                    'gpt-3.5-turbo-16k',
                                                                                                    'gpt-4',
                                                                                                    'llama2'))
                    temp = st.slider("LLM temp", min_value=0.0, max_value=2.0, value=1.0, step=0.1)

                    llm_form = st.form_submit_button('Save')

                if llm_form:
                    load_llm_settings(llm_current, temp)

            if filter_:
                disabled = True
            else:
                disabled = False
            with st.expander("Embedding"):
                # This has to load existing embedding model from local
                with st.form("Embedding settings", border=False):
                    embed_model = st.selectbox('Embedding Model', placeholder='Select an embedding model',
                                               options=('sentence-transformers/all-mpnet-base-v2',
                                                        'BAAI/bge-m3',
                                                        'sentence-transformers/all-MiniLM-L6-v2'), disabled=disabled)

                    embed_chunk_size = st.text_input('Chunk size', placeholder="Enter a chunk size", value=256,
                                                     max_chars=3,
                                                     disabled=disabled)
                    embed_chunk_overlap = st.text_input('Chunk overlap', placeholder="Enter chunk overlap", value=25,
                                                        max_chars=2, disabled=disabled)
                    embedding_form = st.form_submit_button('Save', on_click=load_embedding_settings,
                                                           args=(embed_model,),
                                                           disabled=disabled)

                if embedding_form:
                    load_embedding_settings(embed_model)

            with st.expander("Node Parser"):
                with st.form("Node parser settings", clear_on_submit=False, border=False):
                    parser = st.selectbox('Parser', placeholder='Select a node parser',
                                          options=('HTMLNodeParser',
                                                   'JSONNodeParser',
                                                   'MarkdownNodeParser',
                                                   'JSONNodeParser',
                                                   ), disabled=disabled)
                    splitter = st.selectbox('Text splitters', placeholder="Pick the suitable splitter",
                                            options=(
                                                'CodeSplitter',
                                                'SentenceSplitter',
                                                'SentenceWindowNodeParser',
                                                'SemanticSplitterNodeParser',
                                                'TokenTextSplitter',
                                                'HierarchicalNodeParser',
                                            ), disabled=disabled)
                    st.form_submit_button('Save', disabled=disabled)

            with st.expander("Retriever"):
                with st.form("Query settings", clear_on_submit=False, border=False):
                    mode = st.selectbox('Retrieval mode', placeholder='Select an ideal mode of retrieval',
                                        options=('default',
                                                 'embedding',
                                                 'llm'))
                    ret_modules = st.selectbox('Retrievers', ['Hybrid', 'Vector + Keyword', 'Vector',
                                                              'BM25', 'Keyword'
                                                              ], placeholder='Select the retriever/s you want to use',
                                               help="RRF only works for Hybrid mode")
                    rrf_setting = st.toggle("Use RRF to rerank")
                    submit = st.form_submit_button('Save')
                    if submit:
                        set_retrieval_settings(ret_modules, rrf_setting)

            with st.expander("Postprocessor (Filters)"):
                cb1 = st.checkbox("SimilarityPostprocessor", help="Filters out nodes which are strictly greater than "
                                                                  "threshold"
                                                                  " provided")
                if cb1:
                    with st.form("similarity filter", border=False):
                        sim_num = st.number_input("similarity percent", min_value=0.0, max_value=1.0, value=0.5,
                                                  step=0.1)

                        save = st.form_submit_button("Add Filter")

                        if save:
                            update_similaritypostprocessor(sim_num)
                cb2 = st.checkbox("KeywordNodePostprocessor", help="Add allowed and excluded keywords post retrieval to"
                                                                   "filter relevant nodes")
                if cb2:
                    with st.form("keywordprocessor", border=False):
                        allowed_li = st.text_input("Allowed Keywords",
                                                   placeholder="Enter a comma separated list of search keywords")
                        excluded_li = st.text_input("Excluded Keywords", placeholder="Enter a comma separated list of "
                                                                                     "excluded"
                                                                                     "keywords")
                        save = st.form_submit_button("Add Filter")
                        if save:
                            if allowed_li or excluded_li:
                                allowed_list = allowed_li.split(',')
                                excluded_list = excluded_li.split(',')
                                update_keywordNodePostprocessor(allowed_list, excluded_list)

                cb7 = st.checkbox("PIINodePostprocessor", help="Removes potential PII information in the data",
                                  disabled=True)

            with st.expander("Postprocessor (Rerankers)"):

                cb3 = st.checkbox('LongContextReorder', help="In cases where actual context might be in middle, "
                                                             "it reorders the"
                                                             "context", disabled=True)

                cb4 = st.checkbox("SentenceTransformerRerank", help="Rerank using Sentence Transformer")
                if cb4:
                    set_reranker()

                cb5 = st.checkbox("LLM Rerank",
                                  help="Reranks using LLM to return relevant documents and relevancy score",
                                  disabled=True)
                cb6 = st.checkbox("FixedRecencyPostprocessor",
                                  help="Reranks the node based on their recency, needs a date "
                                       "field in metadata", disabled=True)

                cb8 = st.checkbox("Colbert Reranker", help="Uses the Colbert Reranker to rerank the documents",
                                  disabled=True)
                cb9 = st.checkbox("rankLLM", help="Uses rankLLM to rerank the documents", disabled=True)

            btn = st.button("Complete Setup", type="primary")

            if btn:
                print("Creating Query Engine...")
                with st.status("Building Query Engine...") as qr:
                    if retr := st.session_state.retriever:
                        st.write("Retriever Added")
                        print("\n\n\nretriever is: ", retr)
                    if post_processor := st.session_state.reranker:
                        st.write("post processor added")
                        print("\n\n\nreranker is: ", post_processor)

                    if retr is None:
                        st.toast("Please select a Retriever")
                        time.sleep(0.5)
                        st.toast("Please select a Retriever")
                        time.sleep(0.5)
                        st.toast("Please select a Retriever")
                    elif post_processor is None:
                        st.toast("Please pick a postprocessor/reranker")
                        time.sleep(0.5)
                        st.toast("Please pick a postprocessor/reranker")
                        time.sleep(0.5)
                        st.toast("Please pick a postprocessor/reranker")
                    else:
                        response_synthesizer = get_response_synthesizer()
                        st.session_state.queryEngine = RetrieverQueryEngine.from_args(retriever=retr,
                                                                                      node_postprocessors=post_processor,
                                                                                      response_synthesizer=
                                                                                      response_synthesizer)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": f"I'm ready to answer your questions"}
    ]

col1, col2 = st.columns([0.8, 0.2], gap='medium')

if st.session_state.queryEngine:
    disabled = False
else:
    disabled = True
if query := st.chat_input("Your Question", disabled=disabled):
    st.session_state.messages.append({"role": "user", "content": query})

with col2:
    reset_button = st.button("Reset Chat")
    if reset_button:
        st.session_state.messages = [
            {"role": "assistant", "content": f"I'm ready to answer your questions"}
        ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.queryEngine.query(query)
            st.write(response.response)
            with st.expander("") as c:
                for resp in response.source_nodes:
                    st.write("**context**")
                    st.write(resp.text, "\n")
                    st.write("score", resp.score)
                    st.write("")
                    st.write("")
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)
