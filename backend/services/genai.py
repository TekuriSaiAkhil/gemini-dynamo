from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAI
from vertexai.generative_models import GenerativeModel
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
import logging
from tqdm import tqdm
import json

#Configure log
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeminiProcessor:
    def __init__(self,model_name, project) -> None:
        self.model = VertexAI(model_name=model_name, project = project)

    def document_summary(self, documents: list, **args):

        chain_type = "map_reduce" if len(documents) >10 else "stuff"

        chain = load_summarize_chain(llm=self.model, chain_type=chain_type, **args)

        return chain.run(documents)
    
    def count_total_tokens(self, docs: list):
        temp_model =  GenerativeModel("gemini-1.0-pro")
        total = 0
        logger.info("Counting total billable characters...")
        
        for doc in tqdm(docs):
            total += temp_model.count_tokens(doc.page_content).total_billable_characters
        
        return total

    def get_model(self):
        return self.model


class YoutubeProcessor:

    def __init__(self, genai_processor: GeminiProcessor) -> None:
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=0)
        self.GeminiProcessor = genai_processor
    
    def retrive_youtube_documents(self, video_url:str, verbose = False):
        loader  = YoutubeLoader.from_youtube_url(video_url, add_video_info=True)
        docs = loader.load()
        result = self.text_splitter.split_documents(docs)

        author = result[0].metadata['author']
        length = result[0].metadata['length']
        title = result[0].metadata['title']
        total_size = len(result)
        total_billable_characters = self.GeminiProcessor.count_total_tokens(result)

        if verbose:
            logger.info(f"{author}\n{length}\n{title}\n{total_size}\n{total_billable_characters}")

        return result
    
    def get_key_concepts(self, documents: list, sample_size: int =0, verbose=False):
        # iterate over documents of key size N and find key concepts
        if sample_size > len(documents):
            raise ValueError("Group size is larger than the number of documents")

        #Optimize sample size given no input
        if sample_size==0:
            sample_size = len(documents)//5
        
        num_docs_of_group = len(documents)// sample_size + (len(documents)%sample_size > 0)
        if verbose:
            logging.info(f"No sample size specified. Setting the number of docs per group to 5. Sample size: {sample_size}")

        #Check thresholds for our response quality
        if num_docs_of_group > 10:
            raise ValueError("Each group has more than 10 documents and output quality will be significantly degraded. Increase the sample_size to reduce the number of documents in a group.")
        elif num_docs_of_group > 5:
            logging.info("Each group has 5 documents and output quality is likely to be degraded. Consider increaseing sample_size")


        groups = [documents[i:i+num_docs_of_group] for i in range(0, len(documents), num_docs_of_group)]

        batch_concepts = []
        batch_cost = 0.0
        logger.info("Finding the key concepts")

        for group in tqdm(groups):
            group_content = ""

            for doc in group:
                group_content += doc.page_content

            prompt = PromptTemplate(
                    template=""" 
                    Find the key concepts or terms found in the text:
                    {text}

                    Respond in the following format as a valid JSON object without any backticks, seperating each concept with a comma:
                    {{"concept": "definition", "concept": "definition", ...}}
                    """, input_variables=["text"])
            chain = prompt | self.GeminiProcessor.model

            output_concept = chain.invoke({"text": group_content})
            batch_concepts.append(output_concept)

            if verbose:
                total_input_char = len(group_content)
                total_input_cost = (total_input_char/1000) * 0.000125

                logging.info(f"Running chain on {len(group)} documents")
                logging.info(f"Total input characters: {total_input_char}")
                logging.info(f"Total cost: {total_input_cost}")

                total_output_char = len(output_concept)
                total_output_cost = (total_output_char/1000) * 0.000375
                
                logging.info(f"Total output characters: {total_output_char}")
                logging.info(f"Total cost: {total_output_cost}")

                batch_cost  += (total_input_cost+ total_output_cost)

                logging.info(f"Total group cost: {total_input_cost+ total_output_cost}\n")

        processed_concepts = []
        for batch_concept in batch_concepts:
            batch_concept = batch_concept.replace("```json", "").replace("```", "").strip()
            start = batch_concept.index('{')
            end = batch_concept.rindex('}')
            batch_concept = batch_concept[start:end+1]
            # print(batch_concept)
            processed_concepts.append(json.loads(batch_concept))

        logging.info(f"Total analysis cost: ${batch_cost}")
        return processed_concepts

