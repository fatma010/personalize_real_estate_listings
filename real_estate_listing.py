import openai
#pip install --upgrade attrs
#pip install openai==0.28 # use when generating real estate list
pip install langchain
pip install langchain_openai
pip install langchain_community
pip install chromadb

# Define your OpenAI API key
if os.environ.get("OPENAI_API_KEY") is None or os.environ.get("OPENAI_API_KEY") =="":
    print("open_ai_api_key is not set as environment variable")
else:
    print("Open AI API Key is set")

#get open_ai_api_key
api_key= os.environ.get("OPEN_AI_API_KEY")

# Create variables to store real estate inputs
neighborhood_name ="Green Oaks"
price ="500,000"
bedroom_num ="3"
bathroom_num = "2"
house_size_sqft="2,000"

description ="Welcome to this eco-friendly oasis nestled in the heart of Green Oaks. This charming 3-bedroom, 2-bathroom home boasts energy-efficient features such as solar panels and a well-insulated structure. Natural light floods the living spaces, highlighting the beautiful hardwood floors and eco-conscious finishes. The open-concept kitchen and dining area lead to a spacious backyard with a vegetable garden, perfect for the eco-conscious family. Embrace sustainable living without compromising on style in this Green Oaks gem."
neighborhood_description="Green Oaks is a close-knit, environmentally-conscious community with access to organic grocery stores, community gardens, and bike paths. Take a stroll through the nearby Green Oaks Park or grab a cup of coffee at the cozy Green Bean Cafe. With easy access to public transportation and bike lanes, commuting is a breeze."

#prompts for generating real estate lists
prompt = "Create real estate listings, focusing on neighbourhood, price size and number of rooms. Highlight description and neighboorhood description especially. "
prompt_template = f"Generate real estate listings, you can focus on different features of real estates for example {neighborhood_name}, $ {price},{bedroom_num},{bathroom_num}, sqft {house_size_sqft}.Highlight both {description} and {neighborhood_description} especially."
print(prompt_template)

# Function to call the OpenAI GPT-3.5 API for generating real estate listings
def generate_real_estate_listings(prompt_template):
    try:
        # Calling the OpenAI API with a system message and our prompt in the user message content
        # Use openai.ChatCompletion.create for openai < 1.0
        # openai.chat.completions.create for openai > 1.0
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",

            messages=[
                {
                    "role": "system",
                    "content": "You are a real estate creator. You generate real estate lists according to preferences. Create sentences according to max_tokens parameter,generate complete sentences "
                },
                {
                    "role": "user",
                    "content": prompt_template
                }
            ],
            temperature=1,
            # engine="text-davinci-003",
            max_tokens=150, #more token gives better results, however use an optimized number bec of the cost
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        # The response is a JSON object containing more information than the generated review. We want to return only the message content
        return response.choices[0].message.content
        # return response.choices[0].message.strip()
    except Exception as e:
        return f"An error occurred: {e}"


# Generate response from the model
real_estates = generate_real_estate_listings(prompt_template)

# Printing the output.
print("Generated real_estate:")
print(real_estates)

prompt = "Generate real estate list for customers, you can mention different characteristics for instance environment, location and number of rooms"

#generate 10 example real estate listings as semantic data and print them
listings = [generate_real_estate_listings(prompt) for _ in range(10)]
#print(listings[9])
#print(listings)

#split real estates listings and create chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter
chunk_size=100
chunk_overlap=10
splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size,
                                         chunk_overlap=chunk_overlap)
text = [splitter.split_text(listings[i]) for i in range(10)]

#embed real estate listings chunks and store them as embeds in Chroma vector db
from langchain_community.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
# create the open-source embedding function
embedding_function = OpenAIEmbeddings(openai_api_key=api_key)
# load them into Chroma
vectordb = Chroma(#persist_directory = persist_directory,
                 embedding_function = embedding_function)
vectordb.persist()
for x in range(10):
    textstorage = Chroma.from_texts(text[x], embedding_function)

#get user preferences
buyer_preference = "Looking for a 2-bedroom house near the city center"
#Preference_1 = "Looking for a 2-bedroom house with a budget of $500,000"
#Preference_2 = "Looking for a house which one is the most eco-friendly"
# search the most similar real estate listings for the buyer preferences
similar_real_estate = textstorage.similarity_search(buyer_preference)

# print results, there maybe more than one result, get closely match the input preferences.
print(similar_real_estate[0].page_content)
#print(estates)

#another approach for similarity search with similarity score, number of "k" list will return as result
#import chromadb
#embeddings = OpenAIEmbeddings(openai_api_key=api_key)
#new_client = chromadb.EphemeralClient()
#openai_lc_client = Chroma.from_documents(
#    docs, embeddings, client=new_client, collection_name="openai_collection")

#query = "Looking for a 2-bedroom house near the city center"
#query = "Looking for a 2-bedroom house with a budget of $500,000"
#sim_real_est =openai_lc_client.similarity_search_with_score(query,k=3)
#print(sim_real_est)
#print(sim_real_est[0][0])

#print(sim_real_est[2][0].page_content)

#get preferences from customers and use them for personalization

#more than one seacrh result can be used, as an example use the most similar search result for personalization
matched_listings = [similar_real_estate[0].page_content]
#print(buyer_preferences)
#print(matched_listings)

#reproduce real estate listing according to buyer preferences

def augment_listing(listing, preference):
    prompt = f"Personalize the following real estate lists based on the buyer's preference: {buyer_preferences}\n\nlisting: {matched_listings}."
    response = openai.Completion.create(
        #engine="davinci-002",
        engine ="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=150
        
    )
    return response.choices[0].text
#response.choices[0].text.strip()


personalized_listing = [augment_listing(matched_listings, buyer_preferences) for listing in matched_listings]
print(personalized_listing[0].strip())











