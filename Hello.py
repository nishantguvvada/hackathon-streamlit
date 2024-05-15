import streamlit as st
import replicate
import os
from transformers import AutoTokenizer
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)

def run():
    # Set assistant icon to Snowflake logo
    icons = {"assistant": "./Snowflake_Logomark_blue.svg", "user": "⛷️"}

    # App title
    st.set_page_config(page_title="AI-den")

    # Replicate Credentials
    with st.sidebar:
        st.title("Snowflake Arctic's - (AI)den")
        st.caption("Your personal AI-assisted den")
        if 'REPLICATE_API_TOKEN' in st.secrets:
            replicate_api = st.secrets['REPLICATE_API_TOKEN']
        else:
            replicate_api = st.text_input('Enter Replicate API token:', type='password')
            if not (replicate_api.startswith('r8_') and len(replicate_api)==40):
                st.warning('Please enter your Replicate API token.', icon='⚠️')
                st.markdown("**Don't have an API token?** Head over to [Replicate](https://replicate.com) to sign up for one.")

        os.environ['REPLICATE_API_TOKEN'] = replicate_api
        option = st.selectbox(
            "Select the mode for use-specific responses",
            ("Homework", "Business", "Curiosity", "Games"),
            index=None,
            placeholder="Select a mode",
        )
        st.write("You selected:", option)
        st.subheader("Adjust model parameters")
        temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.3, step=0.01)
        top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)

    # Display introduction
    st.title(":blue[AI-den] - Your personal den of knowledge!")

    # Store LLM-generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "Hi. I'm AI-den. Ask me anything."}]

    # Display or clear chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=icons[message["role"]]):
            st.write(message["content"])

    def clear_chat_history():
        st.session_state.messages = [{"role": "assistant", "content": "Hi. I'm AI-den. Ask me anything."}]

    st.sidebar.button('Clear chat history', on_click=clear_chat_history)
    st.sidebar.caption('Built by [Nishant Guvvada](https://www.linkedin.com/in/nishant-guvvada-36647289/) to demonstrate [Snowflake Arctic](https://www.snowflake.com/blog/arctic-open-and-efficient-foundation-language-models-snowflake). App hosted on [Streamlit Community Cloud](https://streamlit.io/cloud). Model hosted by [Replicate](https://replicate.com/snowflake/snowflake-arctic-instruct).')

    @st.cache_resource(show_spinner=False)
    def get_tokenizer():
        """Get a tokenizer to make sure we're not sending too much text
        text to the Model. Eventually we will replace this with ArcticTokenizer
        """
        return AutoTokenizer.from_pretrained('Snowflake/snowflake-arctic-embed-l')
        # AutoTokenizer.from_pretrained("Snowflake/snowflake-arctic-instruct", trust_remote_code=True, revision="refs/pr/3")
        # AutoTokenizer.from_pretrained("snowflake/snowflake-arctic-instruct", trust_remote_code=True) 

    def get_num_tokens(prompt):
        """Get the number of tokens in a given prompt"""
        tokenizer = get_tokenizer()
        tokens = tokenizer.tokenize(prompt)
        return len(tokens)

    # Function for generating Snowflake Arctic response
    def generate_arctic_response():
        prompt = []
        for dict_message in st.session_state.messages:
            if dict_message["role"] == "user":
                if option == "Homework":
                    prompt.append("Answer in atleast 500 words in a structured format consisting of sections named 'Introduction', 'What', 'Why', 'How', 'Advantages' and 'Disadvantages'\n")
                elif option == "Business":
                    prompt.append("The user is a business owner. Answer in atleast 200 in a structured format consisting of sections named 'Introduction', 'What', 'Why', 'How', 'Feasibility', 'Pros' and 'Cons' to give the business owner insights and make a business decision. Include any possible legal complications.\n")
                elif option == "Curiosity":
                    prompt.append("Answer as if explaining to a 5th grade student, with examples\n")
                elif option == "Games":
                    prompt.append("Answer in 10 words. In the next line, add a quiz question related to the answer.\n")
                else:
                    prompt.append("\n")
                prompt.append("<|im_start|>user\n" + dict_message["content"] + "<|im_end|>")
            else:
                prompt.append("<|im_start|>assistant\n" + dict_message["content"] + "<|im_end|>")
        
        prompt.append("<|im_start|>assistant")
        prompt.append("")
        prompt_str = "\n".join(prompt)
        
        if get_num_tokens(prompt_str) >= 3072:
            st.error("Conversation length too long. Please keep it under 3072 tokens.")
            st.button('Clear chat history', on_click=clear_chat_history, key="clear_chat_history")
            st.stop()

        for event in replicate.stream("snowflake/snowflake-arctic-instruct",
                              input={"prompt": prompt_str,
                                      "prompt_template": r"{prompt}",
                                      "temperature": temperature,
                                      "top_p": top_p,
                                      }):
            yield str(event)

    # User-provided prompt
    if prompt := st.chat_input(disabled=not replicate_api):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="⛷️"):
            st.write(prompt)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant", avatar="./Snowflake_Logomark_blue.svg"):
            response = generate_arctic_response()
            full_response = st.write_stream(response)
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)



if __name__ == "__main__":
    run()
