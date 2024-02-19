""" Analyst base class. """

from abc import ABCMeta, abstractmethod
from typing import Any, Dict


class Analyst(metaclass=ABCMeta):
    """ Base Analyst class. """

    def __init__(self):
        """ Initializes Analyst objects. """

    @abstractmethod
    def query_analyst(self, user_input: str) -> str:
        """ Queries the analyst. """
        raise NotImplementedError

    def launch_chat_box(self):
        """
        https://medium.com/@manirajudutha16/building-a-chatbot-with-openai-and-adding-a-gui-with-tkinter-in-python-602c4a803bcc
        """
        import tkinter as tk
        from tkinter import scrolledtext, END

        # Function to display the chatbot response in the GUI
        def show_chatbot_response():
            user_input = user_input_box.get("1.0", END).strip()
            user_input_box.delete("1.0", END)

            if user_input.lower() in ["exit", "quit", "bye"]:
                chat_log.insert(tk.END, "Chatbot: Goodbye!\n")
                return

            chat_log.insert(tk.END, "You: " + user_input + "\n")
            chatbot_response = self.query_analyst(user_input)
            chat_log.insert(tk.END, "Chatbot: " + chatbot_response + "\n")

        # Main GUI window
        root = tk.Tk()
        root.title("Chatbot with OpenAI")

        # Chat log
        chat_log = scrolledtext.ScrolledText(root, width=60, height=20, wrap=tk.WORD)
        chat_log.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        # User input box
        user_input_box = scrolledtext.ScrolledText(root, width=40, height=4, wrap=tk.WORD)
        user_input_box.grid(row=1, column=0, padx=10, pady=10)

        # Send button
        send_button = tk.Button(root, text="Send", command=show_chatbot_response)
        send_button.grid(row=1, column=1, padx=10, pady=10)

        # Start the GUI event loop
        root.mainloop()