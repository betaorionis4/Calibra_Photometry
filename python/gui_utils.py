import tkinter as tk
from tkinter import ttk

def add_copy_context_menu(widget):
    """
    Adds a standard right-click context menu (Copy, Select All) to a widget.
    Supports tk.Text, tk.Entry, and ttk.Entry.
    """
    menu = tk.Menu(widget, tearoff=0)
    
    def copy_text():
        try:
            # Check if there is a selection
            text = ""
            if isinstance(widget, tk.Text):
                if widget.tag_ranges(tk.SEL):
                    text = widget.get(tk.SEL_FIRST, tk.SEL_LAST)
            elif isinstance(widget, (tk.Entry, ttk.Entry)):
                if widget.selection_present():
                    text = widget.get()[widget.index(tk.SEL_FIRST):widget.index(tk.SEL_LAST)]
            
            if text:
                widget.clipboard_clear()
                widget.clipboard_append(text)
        except tk.TclError:
            pass # No selection

    def select_all():
        if isinstance(widget, tk.Text):
            widget.tag_add(tk.SEL, "1.0", tk.END)
        elif isinstance(widget, (tk.Entry, ttk.Entry)):
            widget.selection_range(0, tk.END)
            widget.icursor(tk.END)

    menu.add_command(label="Copy (Ctrl+C)", command=copy_text)
    menu.add_command(label="Select All (Ctrl+A)", command=select_all)

    def show_menu(event):
        # Try to focus the widget first
        try:
            widget.focus_set()
        except:
            pass
        menu.post(event.x_root, event.y_root)

    # Bind right-click (Button-3 on Windows/Linux)
    widget.bind("<Button-3>", show_menu)

class SelectableLabel(tk.Entry):
    """
    A widget that looks and behaves like a tk.Label but allows the user 
    to highlight text with the mouse and copy it.
    """
    def __init__(self, parent, textvariable=None, text="", font=("Arial", 9), justify=tk.LEFT, fg=None, bg=None, padx=0, pady=0, **kwargs):
        if textvariable is None:
            self._internal_var = tk.StringVar(value=text)
            textvariable = self._internal_var
            
        # Try to match the parent's background for a seamless look if bg not provided
        if bg is None:
            try:
                bg = parent.cget("bg")
            except:
                bg = "#f0f2f5" 
        
        # Default foreground if not provided
        if fg is None:
            fg = "#333333"

        super().__init__(parent, textvariable=textvariable, state="readonly", 
                         readonlybackground=bg, fg=fg, relief="flat", borderwidth=0, 
                         highlightthickness=0, font=font, justify=justify, **kwargs)
        
        # Handle padding for pack/grid if passed
        self.padx = padx
        self.pady = pady
        
        # Add the context menu automatically
        add_copy_context_menu(self)

def add_treeview_copy_menu(tree):
    """
    Adds a right-click menu to a Treeview to copy data formatted for Excel (tab-separated).
    """
    menu = tk.Menu(tree, tearoff=0)
    
    def copy_to_clipboard(iids):
        if not iids:
            return
        
        lines = []
        # Header (Column titles)
        cols = tree["columns"]
        header = "\t".join([tree.heading(c)["text"] for c in cols if tree.heading(c)["text"]])
        if header:
            lines.append(header)
        
        # Row Data
        for iid in iids:
            values = tree.item(iid, "values")
            lines.append("\t".join([str(v) for v in values]))
            
        data = "\n".join(lines)
        tree.clipboard_clear()
        tree.clipboard_append(data)

    def copy_selected():
        copy_to_clipboard(tree.selection())

    def copy_all():
        copy_to_clipboard(tree.get_children())

    menu.add_command(label="Copy Selected Rows (Excel-ready)", command=copy_selected)
    menu.add_command(label="Copy Entire List (Excel-ready)", command=copy_all)

    def show_menu(event):
        # If user clicks a row that isn't selected, select it
        item = tree.identify_row(event.y)
        if item and item not in tree.selection():
            tree.selection_set(item)
            
        menu.post(event.x_root, event.y_root)

    tree.bind("<Button-3>", show_menu)
