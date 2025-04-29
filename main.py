# Import the libraries
import pandas as pd
from IPython.display import display
from get_chat_completions import get_chat_completions
from initialize_conversation import initialize_conversation
from moderation_check import moderation_check
from intent_confirmation_layer import intent_confirmation_layer

df = pd.read_csv('laptop_data.csv')

debug_conversation = initialize_conversation()
debug_conversation.append({"role": "user", "content": "Hi, I am Anshuman. I need a laptop for coding."})

debug_response_assistant = get_chat_completions(debug_conversation)
debug_conversation.append(({"role": "assistant", "content": debug_response_assistant}))

print(moderation_check(debug_response_assistant))
print(intent_confirmation_layer(debug_response_assistant))

