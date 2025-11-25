# eceeasy_advanced.py (Final Corrected Version with Chunking and Prompt Refinements)
import os
import json
import re
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import shutil

# -------------------------------------------------------
#  SETUP
# -------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB = os.path.join(BASE_DIR, "chroma_db")
COURSES_DATA_DIR = os.path.join(BASE_DIR, "courses_data")

# Configuration
# FIX 1: Increased CHUNK_SIZE and CHUNK_OVERLAP for better context capture
CHUNK_SIZE = 700 
CHUNK_OVERLAP = 70
RETRIEVAL_K = 3

print("🔍 Loading Ollama embedding model...")
embedding_model = OllamaEmbeddings(model="llama3") 

# -------------------------------------------------------
#   SIMPLE QUERY CONSTRUCTION (Unchanged)
# -------------------------------------------------------
class CourseMatcher:
    """Simple course matching system"""
    
    def __init__(self, courses):
        self.courses = courses
        self.course_names = [course['name'].lower() for course in courses]
        
    def find_course(self, query):
        """Find the best matching course for a query"""
        query_lower = query.lower()
        
        # Direct matches
        for course_name in self.course_names:
            if course_name in query_lower:
                return course_name
        
        # Keyword matches
        keyword_map = {
            'communication': 'fundamentals of communication system',
            'signal processing': 'digital signal processing', 
            'dsp': 'digital signal processing',
            'vlsi': 'vlsi design',
            'embedded': 'embedded systems design',
            'wireless': 'wireless communication',
            'network': 'network theory',
            'digital electronics': 'digital electronics/ digital logic and systems',
            'signals': 'signals and systems',
            'electromagnetic': 'electromagnetic theory',
            'semiconductor': 'semiconductor devices and applications',
            'rf simulation': 'rf simulation techniques', 
            'passive microwave': 'design of passive microwave components' 
        }
        
        # Find the full course name (case-insensitive)
        for keyword, course_name in keyword_map.items():
            if keyword in query_lower:
                return course_name.lower() # Return lowercase for filtering
        
        return None

# -------------------------------------------------------
#  COURSE DATA MANAGEMENT (Unchanged)
# -------------------------------------------------------
def load_courses_from_json(json_path):
    """Load course data from JSON file"""
    
    if not os.path.exists(json_path):
        print(f"❌ JSON file not found: {json_path}")
        return []
    
    print(f"📖 Loading courses from JSON: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        courses = json.load(f)
    
    print(f"📊 Loaded {len(courses)} courses from JSON")
    return courses

def load_courses_from_directory(data_dir):
    """Load all course data from the courses_data directory"""
    
    if not os.path.exists(data_dir):
        print(f"❌ Courses data directory not found: {data_dir}")
        print("📁 Creating directory structure...")
        os.makedirs(data_dir)
        create_sample_courses_file(data_dir)
        return []
    
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    if not json_files:
        print("❌ No JSON files found in courses_data directory")
        create_sample_courses_file(data_dir)
        return []
    
    all_courses = []
    for json_file in json_files:
        json_path = os.path.join(data_dir, json_file)
        courses = load_courses_from_json(json_path)
        all_courses.extend(courses)
    
    print(f"🎯 Total courses loaded: {len(all_courses)}")
    return all_courses

def create_sample_courses_file(data_dir):
    """Create a sample courses JSON file"""
    # NOTE: Sample courses are used here for completeness. 
    # In a real environment, this function would handle initial data population.
    sample_courses = [
        {
            "name": "Design of Passive Microwave Components",
            "instructor": "prof. Darshak Bhatt",
            "difficulty": "3 out of 5",
            "usefulness": "5 out of 5",
            "challenges": "Cadence workflow, layout steps, simulation setup and design validation",
            "overview": "Design and simulation of passive microwave components and basic MOS/analog circuits using Cadence, including layout practice.",
            "topics": "Common source amplifer,Ring oscillator,low Noice Amplifier",
            "review by students": [
                "This course provides invaluable **hands-on experience in analog VLSI design** using the industry standard, **Cadence software**. The initial struggle with the **Cadence workflow** and unfamiliar tools is the main difficulty, but Prof. Bhatt's guidance is excellent.",
                "The 5/5 usefulness is well-deserved; this is essential for Analog/RF IC design. We learned how to design and simulate core blocks like the **Common Source Amplifier** and the **Ring Oscillator**. **Layout practice** was challenging—it takes time to build confidence and use the tool effectively.",
                "The difficulty rating (3/5) is fair; the concepts are foundational, but the **simulation setup** and achieving **design validation** that matches specifications takes patience. You need to be familiar with **Analog Circuits** and **SPICE concepts** beforehand.",
                "Learning the entire **layout process** for components like the **Low Noise Amplifier (LNA)** was the biggest takeaway. The course teaches you why a schematic simulation differs from post-layout results, which is critical. Highly recommended if you want tangible VLSI skills.",
                "This course effectively bridges theory and application. As the previous review states, **learning a new software is challenging**, and you must dedicate time to practice the meticulous **layout steps**. If you're targeting IC design, this course provides the core software competency."
            ],
            "prerequisites": "Yes, it requires basic knowledge of electronic circuits, semiconductor devices, and circuit analysis.Familiarity with tools like SPICE or basic simulation concepts is also helpful,Analog circuits ",
            "credits": "2",
            "course_code": "TEB:ECT-102  ",
            "semester": "2025 Atumn"
        },
        {
            "name": "RF Simulation Techniques",
            "instructor": "prof. Amalendu Patnaik",
            "difficulty": "3/5 ",
            "usefulness": "4/5",
            "challenges": "3D EM setup, antenna optimization, S-parameter interpretation and simulation convergence",
            "overview": "CST and antenna simulation focus with variable depth,essential for RF design and electromagnetic analysis.Practical electromagnetic simulation using CST Studio Suite for antennas, filters and high-frequency components; emphasizes iterative optimization and analysis",
            "topics": "Ring resonator,Dipole and Monopole Antenna,microchip patch Antenna,stripline and Microstripline,Annalysis of s-parameters",
            "review by students": [
                "This course is highly practical and application-oriented. Prof. Patnaik focuses heavily on hands-on sessions with **CST Studio Suite**. The content isn't conceptually new (as stated, it relies on EM Theory), but learning the **3D EM setup** and ensuring **simulation convergence** is where the real challenge lies. Essential for anyone aiming for an RF/Microwave career.",
                "Excellent course for building skills in **CAD software** for RF components. Modeling the **microstrip patch antenna** and optimizing its performance (gain, bandwidth) was the most rewarding part. The interpretation of **S-parameters** (especially S11 and S21) is critical and well-taught. The deadlines were manageable, making the learning curve smooth.",
                "The difficulty rating of 3/5 is fair, as the complexity is in the software, not the core theory. I really appreciated the focus on practical structures like **ring resonators** and **stripline/microstripline** transmission lines. This course is a direct pathway to industry skills in RF engineering.",
                "Prof. Patnaik provides a great introduction to the workflow of an RF designer. While we learned how to model different antennas, the true value was in understanding how to perform **iterative optimization** to meet specific design targets. Highly recommend if you want tangible skills to put on your resume.",
                "The course is worth it if you are interested in the high-frequency domain. The lectures on the **analysis of S-parameters** were very clear. The main challenge was debugging the **3D EM setup** when unexpected results occurred, but this is a real-world skill the course helps you develop."
            ],
            "prerequisites": "Electromagnetic Theory ",
            "credits": "2",
            "course_code": "TEB:ECT-101",
            "semester": "2025 Spring"
        },
        # ... (other courses remain the same)
    ]
    
    sample_path = os.path.join(data_dir, "ece_courses.json")
    with open(sample_path, 'w', encoding='utf-8') as f:
        json.dump(sample_courses, f, indent=2)
    
    print(f"📝 Created sample course file: {sample_path}")


# -------------------------------------------------------
# CREATE COURSE DOCUMENTS (Handles list/string reviews)
# -------------------------------------------------------
def create_course_documents(courses):
    """Create simple, clear course documents"""
    
    print("📚 Creating course documents...")
    
    if not courses:
        print("❌ No courses found.")
        return []
    
    documents = []

    for course in courses:
        course_name = course['name']
        instructor = course['instructor']
        
        # FIX: Handle 'review by students' being a list or a string
        reviews = course.get('review by students', 'No student reviews available')
        if isinstance(reviews, list):
            # Join list elements into a single, comprehensive string, separated by new lines
            formatted_reviews = "\n- ".join(reviews)
            formatted_reviews = f"- {formatted_reviews}" if reviews else reviews
        elif isinstance(reviews, str):
            formatted_reviews = reviews
        else:
            formatted_reviews = 'No student reviews available'
        
        # Create a comprehensive document for each course
        content = f"""
        COURSE NAME: {course_name}
        INSTRUCTOR: {instructor}
        PROFESSOR: {instructor}
        
        OVERVIEW: {course.get('overview', 'No overview available')}
        DIFFICULTY: {course.get('difficulty', 'Not specified')}
        USEFULNESS: {course.get('usefulness', 'Not specified')}
        CHALLENGES: {course.get('challenges', 'Not specified')}
        TOPICS: {course.get('topics', 'No topics listed')}
        REVIEW BY STUDENTS: 
{formatted_reviews}
        PREREQUISITES: {course.get('prerequisites', 'None')}
        CREDITS: {course.get('credits', 'Not specified')}
        COURSE_CODE: {course.get('course_code', 'N/A')}
        SEMESTER: {course.get('semester', 'N/A')}
        
        This course {course_name} is taught by {instructor}.
        If you want to know who teaches {course_name}, the answer is {instructor}.
        The professor for {course_name} is {instructor}.
        {instructor} is the instructor of {course_name}.
        """.strip()
        
        doc = Document(
            page_content=content,
            metadata={
                "course": course_name,
                "instructor": instructor,
                "course_lower": course_name.lower() # The filter key
            }
        )
        documents.append(doc)
    
    print(f"📊 Created {len(documents)} documents for {len(courses)} courses")
    return documents

# -------------------------------------------------------
# VECTOR STORE SETUP (Uses new CHUNK_SIZE/OVERLAP)
# -------------------------------------------------------
def setup_vector_store(courses):
    """Create vector store"""
    
    if os.path.exists(CHROMA_DB):
        print("🔄 Removing existing database...")
        shutil.rmtree(CHROMA_DB)
    
    print("✨ Creating new vector database...")
    
    documents = create_course_documents(courses)
    
    if not documents:
        print("❌ No documents created.")
        return None
    
    # Split into chunks using the updated size/overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"🧩 Split into {len(chunks)} chunks")
    
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_DB
    )
    vector_store.persist()
    
    return vector_store

# -------------------------------------------------------
# RAG CHAIN SETUP (MODIFIED: Prompt for better list generation/synthesis)
# -------------------------------------------------------
def setup_rag_chain(vector_store, course_filter=None):
    """Setup RAG chain with optional metadata filtering"""
    
    llm = Ollama(model="llama3")
    
    # MODIFIED PROMPT: Emphasize extracting ALL relevant details for a comprehensive summary.
    prompt = ChatPromptTemplate.from_template("""
You are an expert ECE Course Assistant. Answer the question using ONLY the provided CONTEXT.
Provide a comprehensive answer that extracts ALL relevant details. If the question asks for a summary, synthesize all known fields (instructor, overview, difficulty, usefulness, challenges, topics, reviews) from the context into a clear, complete, and easy-to-read response.

CONTEXT:
{context}

QUESTION: {input}

Answer directly and specifically. If the information is not present in the CONTEXT, you MUST state that you do not know.
""")
    
    # --- METADATA FILTERING LOGIC ---
    search_kwargs = {"k": RETRIEVAL_K}
    if course_filter:
        print(f"⚙️ Applying filter to search for course: '{course_filter}'") 
        search_kwargs["filter"] = {"course_lower": course_filter}

    retriever = vector_store.as_retriever(
        search_kwargs=search_kwargs
    )
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)
    
    return rag_chain

# -------------------------------------------------------
#  MAIN APPLICATION (Unchanged)
# -------------------------------------------------------
def main():
    print("🚀 Starting ECEEasy - Course Assistant")
    print("=" * 50)
    
    # Load courses
    courses = load_courses_from_directory(COURSES_DATA_DIR)
    if not courses:
        print("❌ No courses found.")
        return
    
    # Setup course matcher
    course_matcher = CourseMatcher(courses) 
    
    # Setup vector store (This step now indexes the student reviews correctly)
    vector_store = setup_vector_store(courses)
    if not vector_store:
        print("❌ Failed to setup vector store.")
        return
    
    print("\n" + "=" * 50)
    print("🤖 ECEEasy Assistant Ready!")
    print("=" * 50)
    print("📚 Available courses:")
    
    for course in courses:
        print(f"   • {course['name']}")
    
    print(f"\n💡 Ask me about instructors,difficulty,usefulness,challenges,overview,topics,review by students,prerequisites,credicts of the respective course!")
    print()
    
    while True:
        try:
            query = input("🎓 You: ").strip()
            
            if query.lower() in ["exit", "quit"]:
                print("👋 Goodbye!")
                break
            elif not query:
                continue
            
            print("🔍 Searching...")

            # 1. Use the CourseMatcher to find a definite course name
            course_name_lower = course_matcher.find_course(query)
            
            # 2. Setup the RAG chain, passing the course name as a filter if found
            rag_chain = setup_rag_chain(vector_store, course_filter=course_name_lower)
            
            # 3. Execute RAG
            result = rag_chain.invoke({"input": query})
            print(f"\n📘 ECEEasy: {result['answer']}\n")
            
            print("-" * 50)
            print()
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please ensure your Ollama instance is running and try again.\n")

if __name__ == "__main__":
    main()