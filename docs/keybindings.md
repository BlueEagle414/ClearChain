# ClearChain Key Bindings & TUI Navigation

ClearChain features a streamlined Terminal User Interface (TUI) built with the Textual framework. To keep your workflow fast and entirely keyboard-driven, we use the following global key bindings.

## Global Shortcuts

| Key | Action | Description |
| :--- | :--- | :--- |
| **`q`** | **Quit** | Gracefully shuts down the ClearChain daemon and exits the application. |
| **`c`** | **Clear Screen** | Clears the markdown viewer and the real-time log pane to give you a fresh workspace. |
| **`w`** | **Wipe DB** | Opens the Database Wipe confirmation modal. |
| **`a`** | **Add Data** | Opens the Knowledge Base entry modal to manually insert a new Technical Specification. |
| **`Esc`** | **Abort Query** | Instantly cancels the active CoVe pipeline query and stops the LLM generation stream. |

## Interactive Modals

Certain actions in ClearChain will open secure, focused modals over your main workspace to prevent accidental data loss.

### The "Wipe DB" Modal (`w`)
Pressing `w` brings up a strict warning dialog. Because ClearChain focuses on data integrity, this action will permanently delete all vector tables (`tech_specs` and `cove_cache`) in your LanceDB instance. 
* Use your mouse or `Tab` key to select **"Yes, Wipe DB"** (highlighted in red) to proceed, or **"Cancel"** to return to the main screen safely.

### The "Add Data" Modal (`a`)
Pressing `a` opens a data entry form allowing you to manually inject context directly into the local vector database.
* **Entity Name:** A short, unique identifier (e.g., "Omega-77").
* **Details:** The actual text data/specifications (limit: 50,000 characters). 
* Select **"Save"** to automatically chunk, embed, and store the text, which will instantly invalidate the local semantic cache to ensure the AI uses your newest data!

## UI Navigation

* **Query Input:** Focus is automatically placed in the bottom query bar on startup.
* **Settings Bar:** You can use your mouse to dynamically adjust the `Sim Thresh` (Similarity) and `Conf Thresh` (Confidence) input fields at the top right of the screen at any time. Press `Enter` after changing a value to apply it instantly.