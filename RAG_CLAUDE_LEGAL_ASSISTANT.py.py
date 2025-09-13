from langchain.schema import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from datetime import datetime
import uuid
import json
import glob
from dotenv import load_dotenv
import torch
from tqdm import tqdm
import time

# Load environment variables
load_dotenv()

# Check GPU availability
print("🔍 Checking GPU availability...")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"✅ GPU detected: {gpu_name}")
    print(f"   Memory: {gpu_memory:.1f} GB")
    
    if gpu_memory <= 4:  # GTX 1650 Ti has 4GB
        print("   ⚠️  Limited VRAM detected - optimizing for 4GB GPU")
        device = "cuda"
        batch_size = 16  # Smaller batches for 4GB VRAM
    else:
        device = "cuda"
        batch_size = 32
else:
    print("❌ No GPU detected, using CPU")
    device = "cpu"
    batch_size = 8

# Initialize Claude models
llm1 = ChatAnthropic(model="claude-opus-4-1-20250805", temperature=0.9)
llm2 = ChatAnthropic(model="claude-opus-4-1-20250805", temperature=0.9)
llm3 = ChatAnthropic(model="claude-opus-4-1-20250805", temperature=0.9)

# Generate a unique session ID for this run
session_id = str(uuid.uuid4())[:8]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

class CaseLawRAG:
    def __init__(self, json_folder_path="cases_20250617/"):
        self.json_folder_path = json_folder_path
        self.vectorstore = None
        self.device = device
        self.batch_size = batch_size
        
        print(f"🚀 Initializing with {self.device.upper()} acceleration...")
        print(f"   Batch size: {self.batch_size} (optimized for 4GB VRAM)")
        
        # Clear GPU cache before starting
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        # Initialize embeddings without show_progress_bar in encode_kwargs
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",  # Lightweight model for 4GB VRAM
            model_kwargs={'device': self.device},
            encode_kwargs={
                'device': self.device,
                'batch_size': self.batch_size
                # Removed show_progress_bar as it's handled by the class
            }
        )
        # Enable progress bar separately
        self.embeddings.show_progress = True
        self.initialize_vectorstore()
    
    def simplify_metadata(self, metadata):
        """Convert complex metadata values to strings for ChromaDB compatibility"""
        simplified = {}
        for key, value in metadata.items():
            if isinstance(value, (list, dict)):
                # Convert to string but limit length for memory efficiency
                simplified[key] = json.dumps(value)[:500]  # Limit length
            else:
                simplified[key] = str(value)[:200] if value is not None else ""  # Limit length
        return simplified
    
    def load_json_files(self):
        """Load JSON files with memory optimization for 4GB GPU"""
        json_files = glob.glob(f"{self.json_folder_path}*.json")
        documents = []
        
        print(f"📂 Found {len(json_files)} case law files. Loading...")
        
        for file_path in tqdm(json_files, desc="Loading JSON files"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                filename = os.path.basename(file_path)
                
                # Create optimized content - only essential fields
                content_parts = []
                essential_fields = ['Title', 'CaseNumber', 'Content', 'Decisions', 'Opinions']
                
                for field in essential_fields:
                    if field in data and data[field]:
                        # Limit content length to save memory
                        content = str(data[field])
                        if len(content) > 1000:  # Truncate very long content
                            content = content[:1000] + "..."
                        content_parts.append(f"{field}: {content}")
                
                content = "\n".join(content_parts)
                
                # Minimal metadata to save VRAM
                metadata = {
                    "source": filename,
                    "title": str(data.get('Title', 'N/A'))[:50],
                    "case_number": str(data.get('CaseNumber', 'N/A'))[:20],
                }
                
                # Add date if available
                if 'Date' in data and data['Date']:
                    metadata['date'] = str(data['Date'])[:20]
                
                simplified_metadata = self.simplify_metadata(metadata)
                
                documents.append(Document(
                    page_content=content,
                    metadata=simplified_metadata
                ))
                
                # Clear memory periodically
                if len(documents) % 100 == 0 and self.device == "cuda":
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"\n⚠️ Error loading {file_path}: {e}")
                continue
        
        return documents
    
    def initialize_vectorstore(self):
        """Initialize vector store with memory optimization"""
        # Clear GPU memory before starting
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        documents = self.load_json_files()
        
        if not documents:
            print("❌ No documents found.")
            return
        
        print(f"📊 Processing {len(documents)} documents...")
        
        # Use smaller chunks for 4GB VRAM
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Smaller chunks
            chunk_overlap=100
        )
        
        # Process in small batches to avoid VRAM overload
        processing_batch_size = 50  # Smaller batches for processing
        all_split_docs = []
        
        for i in tqdm(range(0, len(documents), processing_batch_size), desc="Splitting documents"):
            batch = documents[i:i + processing_batch_size]
            split_docs = text_splitter.split_documents(batch)
            all_split_docs.extend(split_docs)
            
            # Clear GPU memory after each batch
            if self.device == "cuda":
                torch.cuda.empty_cache()
                time.sleep(0.1)  # Small delay
        
        print(f"🧠 Creating embeddings for {len(all_split_docs)} chunks...")
        
        # Create vector store with progress
        self.vectorstore = Chroma.from_documents(
            documents=all_split_docs,
            embedding=self.embeddings,
            persist_directory="./case_law_db"
        )
        
        # Final memory cleanup
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        print(f"✅ Vector store initialized with {len(all_split_docs)} document chunks")
        print(f"💾 GPU memory usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    def retrieve_relevant_cases(self, query, k=5):
        """Retrieve relevant cases based on the query"""
        if self.vectorstore is None:
            print("Vector store not initialized.")
            return "No case law database available."
        
        # Clear GPU memory before retrieval
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        # Retrieve relevant documents
        relevant_docs = self.vectorstore.similarity_search(query, k=k)
        
        # Format the retrieved information
        context = "RELEVANT CASE LAW PRECEDENTS:\n\n"
        for i, doc in enumerate(relevant_docs):
            context += f"=== Case {i+1} ===\n"
            context += f"Source: {doc.metadata.get('source', 'Unknown')}\n"
            context += f"Title: {doc.metadata.get('title', 'Unknown')}\n"
            context += f"Case Number: {doc.metadata.get('case_number', 'Unknown')}\n"
            
            if 'date' in doc.metadata:
                context += f"Date: {doc.metadata.get('date', 'Unknown')}\n"
            
            context += f"Content Excerpt: {doc.page_content[:400]}...\n\n"
        
        return context

# Initialize RAG system with progress tracking
print("🔄 Starting RAG system initialization...")
print("💡 This will be much faster with GPU acceleration!")
start_time = time.time()

try:
    case_law_rag = CaseLawRAG("cases_20250617/")
    end_time = time.time()
    print(f"⏱️ Initialization completed in {end_time - start_time:.2f} seconds")
    
    # Test GPU acceleration
    if device == "cuda":
        print("🎯 Testing GPU acceleration with a sample query...")
        test_query = "environmental damage mining arbitration"
        test_result = case_law_rag.retrieve_relevant_cases(test_query, k=2)
        print("✅ GPU acceleration working correctly!")
        
except Exception as e:
    print(f"❌ Error during initialization: {e}")
    print("💡 Trying fallback to CPU...")
    device = "cpu"
    case_law_rag = CaseLawRAG("cases_20250617/")


def save_to_markdown(step_name, content, chain_name, user_query):
    """Save output to markdown file with metadata"""
    # Create directory if it doesn't exist
    os.makedirs("markdown_outputs", exist_ok=True)
    
    filename = f"markdown_outputs/{timestamp}_{session_id}_{chain_name}_{step_name}.md"
    
    markdown_content = f"""# {step_name.replace('_', ' ').title()}

## Metadata
- **Chain**: {chain_name}
- **Step**: {step_name}
- **Session ID**: {session_id}
- **Timestamp**: {datetime.now().isoformat()}
- **Original Query**: {user_query}

## Output

{content}

---
*Generated by LLM Chain Processor*
"""
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    print(f"✓ Saved {step_name} output to {filename}")
    return content

# Updated prompts with RAG context
OPPOSITION_PROMPT = ChatPromptTemplate.from_template("""
# Context

You are a paralegal who is advising the lawyers of [plantiff] on the {case_objective} to provide a sound argument using the facts of the background details provided.

# Relevant Case Law
{case_law_context}

# Details

I'm working on a case representing Fenoscadia Limited, a mining company from Ticadia that was operating in Kronos under an 80-year concession to extract lindoro, a rare earth metal. In 2016, Kronos passed a decree that revoked Fenoscadia's license and terminated the concession agreement, citing environmental concerns. The government had funded a study that suggested lindoro mining contaminated the Rhea River and caused health issues, although the study didn't conclusively prove this. Kronos is now filing an environmental counterclaim in the ongoing arbitration, seeking at least USD 150 million for environmental damage, health costs, and water purification.

# Objective

- {case_objective}
- Present your argument in the format of IREAC
- Use sources from litigation, jus mundi and lawnet
- Provide sources and citations to relevant case law
- If you cannot answer the question, write: "I am sorry. I am a lousy AI, bopes"
- Do not make up an answer or give an answer that is not supported by the given context

# Style

Persuasive, structured, tribunal-ready

# Tone

Confident, precise, sharp and professional

# Audience

The arbitral tribunal

# Response

Provide a sound and convincing argument covering all the issues in the case and the applicable relevent rules and their justification, citing relevant case law where appropriate.

Issue

Rule

Explanation of Rule

Application of Rule

Case: {user_input}
""")

OPPOSITION_ARBITRATION = ChatPromptTemplate.from_template("""
# Context

You are an arbitration assistant who is offering advice to the lawyers of [plantiff] on their arguments of {case_objective} using the background, from the details

# Relevant Case Law
{case_law_context}

# Details

I'm working on a case representing Fenoscadia Limited, a mining company from Ticadia that was operating in Kronos under an 80-year concession to extract lindoro, a rare earth metal. In 2016, Kronos passed a decree that revoked Fenoscadia's license and terminated the concession agreement, citing environmental concerns. The government had funded a study that suggested lindoro mining contaminated the Rhea River and caused health issues, although the study didn't conclusively prove this. Kronos is now filing an environmental counterclaim in the ongoing arbitration, seeking at least USD 150 million for environmental damage, health costs, and water purification.

# Objective

- Present your advice in the format of IREAC
- Use sources from litigation, juls mundi and lawnet
- Provide sources and citations to relevant case law
- If you cannot answer the question, write: "I am sorry. I am a lousy AI, bopes"
- Do not make up an answer or give an answer that is not supported by the given context

# Style

Persuasive, structured, tribunal-ready

# Tone

Confident, precise, sharp and professional

# Audience

The arbitral tribunal

# Response

Analyse the argument given, identify the factual and legal weakneses and then provide an improved and convincing argument covering all the issues in the case and the applicable relevent rules and their justification, citing relevant case law where appropriate.

Issue

Rule

Explanation of Rule

Application of Rule

Previous Argument: {previous_argument}
""")

OPPOSITION_COUNTERARGUMENT = ChatPromptTemplate.from_template("""
# Relevant Case Law
{case_law_context}

Objective-
Analyse the arguments from the [opposition] and generate counter arguments for each of their arguments.

R — Response
Produce a 6–8 minute closing statement (800–1,000 words) that:
1. Opening / attention grabber
2. Core themes (2–4 key messages)
3. Fact summary (chronology, but only key highlights)
4. Legal arguments with rebuttals (organized by issue), citing relevant case law
5. Persuasive highlights (fairness, good faith, proportionality)
6. Relief requested
7. Memorable closing line

Tone: Professional, clear, persuasive, suitable for oral delivery. 
Use short sentences and signposting (First, Second, Finally).
Do not invent facts or authorities—only rely on provided materials.
""")

PARALEGAL_PROMPT = ChatPromptTemplate.from_template("""
# Relevant Case Law
{case_law_context}

You are a paralegal who is advising the lawyers of [defendent] on the {case_objective} using the facts of the background details provided.

# Details

I'm working on a case representing Fenoscadia Limited, a mining company from Ticadia that was operating in Kronos under an 80-year concession to extract lindoro, a rare earth metal. In 2016, Kronos passed a decree that revoked Fenoscadia's license and terminated the concession agreement, citing environmental concerns. The government had funded a study that suggested lindoro mining contaminated the Rhea River and caused health issues, although the study didn't conclusively prove this. Kronos is now filing an environmental counterclaim in the ongoing arbitration, seeking at least USD 150 million for environmental damage, health costs, and water purification.

# Objective

- {case_objective}
- Present your argument in the format of IREAC
- Use sources from litigation, juls mundi and lawnet
- Provide sources and citations to relevant case law
- If you cannot answer the question, write: 'I am sorry. I am a lousy AI, bopes'
- Do not make up an answer or give an answer that is not supported by the given context

# Style

Persuasive, structured, tribunal-ready

# Tone

Confident, precise, sharp and professional

# Audience

The arbitral tribunal

# Response

Provide a sound and convincing argument covering all the issues in the case and the applicable relevent rules and their justification, citing relevant case law where appropriate.

Issue

Rule

Explanation of Rule

Application of Rule

Case: {user_input}
""")

PARALEGAL_ARBITRATION = ChatPromptTemplate.from_template("""
# Relevant Case Law
{case_law_context}

# Context

You are an arbitration assistant who is offering advice to the lawyers of [defendent] on their arguments of {case_objective} using the background, from the details

# Details

I'm working on a case representing Fenoscadia Limited, a mining company from Ticadia that was operating in Kronos under an 80-year concession to extract lindoro, a rare earth metal. In 2016, Kronos passed a decree that revoked Fenoscadia's license and terminated the concession agreement, citing environmental concerns. The government had funded a study that suggested lindoro mining contaminated the Rhea River and caused health issues, although the study didn't conclusively prove this. Kronos is now filing an environmental counterclaim in the ongoing arbitration, seeking at least USD 150 million for environmental damage, health costs, and water purification.

# Objective

- Present your advice in the format of IREAC
- Use sources from litigation, juls mundi and lawnet
- Provide sources and citations to relevant case law
- If you cannot answer the question, write: "I am sorry. I am a lousy AI, bopes"
- Do not make up an answer or give an answer that is not supported by the given context

# Style

Persuasive, structured, tribunal-ready

# Tone

Confident, precise, sharp and professional

# Audience

The arbitral tribunal

# Response

Analyse the argument given, identify the factual and legal weakneses and then provide an improved and convincing argument covering all the issues in the case and the applicable relevent rules and their justification, citing relevant case law where appropriate.

Issue

Rule

Explanation of Rule

Application of Rule

Previous Argument: {previous_argument}
""")

PARALEGAL_COUNTERARGUMENT = ChatPromptTemplate.from_template("""
# Relevant Case Law
{case_law_context}

Objective-
Analyse the arguments from the [plantiffs] and generate counter arguments for each of their arguments.

R — Response
Produce a 6–8 minute closing statement (800–1,000 words) that:
1. Opening / attention graber
2. Core themes (2–4 key messages)
3. Fact summary (chronology, but only key highlights)
4. Legal arguments with rebuttals (organized by issue), citing relevant case law
5. Persuasive highlights (fairness, good faith, proportionality)
6. Relief requested
7. Memorable closing line

Tone: Professional, clear, persuasive, suitable for oral delivery. 
Use short sentences and signposting (First, Second, Finally).
Do not invent facts or authorities—only rely on provided materials.
""")

# Create chain components with markdown saving
# Create chain components with markdown saving
def create_chain_step(prompt_template, llm, step_name, chain_name, extra_vars=None, use_rag=False):
    """Create a chain step that saves output to markdown"""
    def step_with_saving(input_data):
        if isinstance(input_data, dict):
            content = input_data
        else:
            content = {"input": input_data}
        
        # Add extra variables if provided
        if extra_vars:
            content.update(extra_vars)
        
        # Add RAG context if requested
        if use_rag:
            # For steps that have user_input, use that for retrieval
            if 'user_input' in content:
                case_law_context = case_law_rag.retrieve_relevant_cases(content['user_input'])
            # For steps that have previous_argument, use that for retrieval  
            elif 'previous_argument' in content:
                case_law_context = case_law_rag.retrieve_relevant_cases(content['previous_argument'])
            # For other steps, use a default query related to the case
            else:
                case_law_context = case_law_rag.retrieve_relevant_cases("mining environmental damage arbitration counterclaim")
            
            content['case_law_context'] = case_law_context
        
        # Execute the chain step
        chain = prompt_template | llm | StrOutputParser()
        result = chain.invoke(content)
        
        # Save to markdown
        query_for_file = content.get('user_input', content.get('input', 'Unknown query'))
        save_to_markdown(step_name, result, chain_name, query_for_file)
        
        return result
    
    return step_with_saving

# Build Chain 1 with markdown saving
def chain1_processor(user_query):
    """Process Chain 1 with markdown output at each step"""
    # Save original query
    save_to_markdown("original_query", user_query, "chain1", user_query)
    
    # Extract case objective from user query
    case_objective = user_query if user_query else "defend against environmental counterclaim"
    
    # Step 1: Opposition Prompt
    step1 = create_chain_step(
        OPPOSITION_PROMPT, 
        llm1, 
        "step1_opposition_paralegal", 
        "chain1",
        {"user_input": user_query, "case_objective": case_objective},
        use_rag=True
    )
    result1 = step1(user_query)
    
    # Step 2: Opposition Arbitration
    step2 = create_chain_step(
        OPPOSITION_ARBITRATION, 
        llm1, 
        "step2_opposition_arbitration", 
        "chain1",
        {"previous_argument": result1, "case_objective": case_objective},
        use_rag=True
    )
    result2 = step2(result1)
    
    # Step 3: opposition counterargument
    step3 = create_chain_step(
    OPPOSITION_COUNTERARGUMENT, 
    llm2, 
    "step3_opposition_counterargument", 
    "chain1",
    {"previous_argument": result2},  # Add previous_argument to content
    use_rag=True
)
    result3 = step3(result2)
    
    return result3

# Build Chain 2 with markdown saving
def chain2_processor(user_query):
    """Process Chain 2 with markdown output at each step"""
    # Save original query
    save_to_markdown("original_query", user_query, "chain2", user_query)
    
    # Extract case objective from user query
    case_objective = user_query if user_query else "defend against environmental counterclaim"
    
    # Step 1: Paralegal Prompt
    step1 = create_chain_step(
        PARALEGAL_PROMPT, 
        llm2, 
        "step1_paralegal_defense", 
        "chain2",
        {"user_input": user_query, "case_objective": case_objective},
        use_rag=True
    )
    result1 = step1(user_query)
    
    # Step 2: Paralegal Arbitration
    step2 = create_chain_step(
        PARALEGAL_ARBITRATION, 
        llm2, 
        "step2_paralegal_arbitration", 
        "chain2",
        {"previous_argument": result1, "case_objective": case_objective},
        use_rag=True
    )
    result2 = step2(result1)
    
    # Step 3: paralegal counterargument
    step3 = create_chain_step(
    PARALEGAL_COUNTERARGUMENT, 
    llm1, 
    "step3_paralegal_counterargument", 
    "chain2",
    {"previous_argument": result2},  # Add previous_argument to content
    use_rag=True
)
    result3 = step3(result2)
    
    return result3

# Main execution function
# Main execution function
def run_application(user_query):
    """Run both chains and save all outputs to markdown"""
    print(f"Starting processing for query: {user_query[:50]}...")
    print(f"Session ID: {session_id}")
    print("-" * 50)
    
    # Run both chains
    chain1_result = chain1_processor(user_query)
    chain2_result = chain2_processor(user_query)
    
    # Add RAG context for the final judge FIRST
    case_law_context = case_law_rag.retrieve_relevant_cases(user_query)
    
    # Create a final balancer that combines both chains (define template AFTER case_law_context is defined)
    CLAUDE_JUDGE = ChatPromptTemplate.from_template(
f"""
# Final Legal Analysis - Session {session_id}

C — Context
You are the Presiding Judge of an arbitral tribunal hearing the dispute between [plantiff] and [defendant]. [case objective]. Do not give empty placeholders in your response. Consider both arguments without any bias and produce the most likely party who will recieve the favourable verdict.

# Relevant Case Law
{case_law_context}

O — Objective
Moderate the hearing firmly, test the persuasiveness of both sides, and deliver a final reasoned award.

S — Style
 • Judicial, structured, probing.
 • Neutral but not passive — actively question the credibility, evidence, and logic of each side.

T — Tone
 • Impartial, authoritative, analytical.
 • Willing to point out weak arguments on either side.

A — Audience
Both parties in arbitration ([plantiff] and [defendant]).

R — Response
 1. Open proceedings with a neutral summary of the dispute.
 2. After each closing statement, ask challenging questions that expose assumptions, evidence gaps, or logical weaknesses.
 3. Objectively assess the relative strength of each side's arguments.
 4. Issue a final award, structured around jurisdiction, admissibility, and merits, explaining not just the conclusion but why certain arguments were more persuasive than others.
 5. State clearly which party wins the case
Label outputs clearly as [Judge]:
""")
    
    # Final balancing step
    final_balancer_chain = CLAUDE_JUDGE | llm3 | StrOutputParser()
    final_analysis = final_balancer_chain.invoke({
        "chain1_output": chain1_result,
        "chain2_output": chain2_result,
        "case_law_context": case_law_context
    })
    
    # Save final analysis
    save_to_markdown("final_arbitration_analysis", final_analysis, "combined", user_query)
    
    # Save final results summary
    final_content = f"""
# Final Legal Analysis - Session {session_id}

## Metadata
- **Session ID**: {session_id}
- **Timestamp**: {datetime.now().isoformat()}
- **Original Query**: {user_query}

## Judge's Final Analysis

{final_analysis}

---
*Generated by Legal Analysis System*
"""
    
    final_filename = f"markdown_outputs/{timestamp}_{session_id}_FINAL_LEGAL_ANALYSIS.md"
    with open(final_filename, 'w', encoding='utf-8') as f:
        f.write(final_content)
    
    print(f"✓ Final results saved to {final_filename}")
    print("=" * 50)
    
    return {
        "chain1": chain1_result,
        "chain2": chain2_result,
        "final_analysis": final_analysis,
        "session_id": session_id,
        "timestamp": timestamp
    }

# Example usage
if __name__ == "__main__":
    user_input = input("Enter your arbitration case details & objective: ")
    if not user_input.strip():
        user_input = "Defend Fenoscadia Limited against Kronos' environmental counterclaim and argue for compensation for wrongful termination of concession agreement"
    
    results = run_application(user_input)
    
    print("\n✅ Legal Processing complete!")
    print(f"Chain 1 (Opposition) output length: {len(results['chain1'])} characters")
    print(f"Chain 2 (Defense) output length: {len(results['chain2'])} characters")
    print(f"Final Analysis length: {len(results['final_analysis'])} characters")
    print(f"Check the 'markdown_outputs' folder for detailed legal documents")