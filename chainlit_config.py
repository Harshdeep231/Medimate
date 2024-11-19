import chainlit as cl

# Function to define settings when the chatbot starts
@cl.on_start
def on_start():
    # Check the theme of the application (light or dark)
    if cl.theme == "dark":
        # Set the logo for dark mode
        cl.set_logo("C:\MediMateAI\public\logo_dark.png")
    else:
        # Set the logo for light mode
        cl.set_logo("C:\MediMateAI\public\logo_light.png")

# Other custom configurations can be added here
