class EventBus:
    def __init__(self, initial_mode=None):
        self.mode_handlers = {}
        self.mode_exit_handlers = {}
        self.global_mode_change_handlers = []
        self.current_mode = initial_mode

    def register_mode_handler(self, mode, handler):
        """Register a handler for a specific mode."""
        if mode not in self.mode_handlers:
            self.mode_handlers[mode] = []
        self.mode_handlers[mode].append(handler)
        return lambda: self.unregister_mode_handler(mode, handler)

    def register_mode_exit_handler(self, mode, handler):
        """Register a handler that runs when exiting a specific mode."""
        if mode not in self.mode_exit_handlers:
            self.mode_exit_handlers[mode] = []
        self.mode_exit_handlers[mode].append(handler)
        return lambda: self.unregister_mode_exit_handler(mode, handler)

    def register_global_mode_change_handler(self, handler):
        """Register a handler that runs on any mode change."""
        self.global_mode_change_handlers.append(handler)
        return lambda: self.unregister_global_mode_change_handler(handler)

    def unregister_mode_handler(self, mode, handler):
        """Remove a registered mode handler."""
        if mode in self.mode_handlers and handler in self.mode_handlers[mode]:
            self.mode_handlers[mode].remove(handler)
            return True
        return False

    def unregister_mode_exit_handler(self, mode, handler):
        """Remove a registered mode exit handler."""
        if mode in self.mode_exit_handlers and handler in self.mode_exit_handlers[mode]:
            self.mode_exit_handlers[mode].remove(handler)
            return True
        return False

    def unregister_global_mode_change_handler(self, handler):
        """Remove a registered global mode change handler."""
        if handler in self.global_mode_change_handlers:
            self.global_mode_change_handlers.remove(handler)
            return True
        return False

    def change_mode(self, new_mode, *args, **kwargs):
        """Change the current mode and trigger appropriate handlers."""
        if new_mode == self.current_mode:
            return False

        old_mode = self.current_mode

        # Run exit handlers for the current mode
        if old_mode in self.mode_exit_handlers:
            for handler in self.mode_exit_handlers[old_mode]:
                handler(new_mode, *args, **kwargs)

        # Update the current mode
        self.current_mode = new_mode

        # Run global mode change handlers
        for handler in self.global_mode_change_handlers:
            handler(old_mode, new_mode, *args, **kwargs)

        # Run handlers for the new mode
        if new_mode in self.mode_handlers:
            for handler in self.mode_handlers[new_mode]:
                handler(*args, **kwargs)

        return True

    def get_current_mode(self):
        """Get the current mode of the system."""
        return self.current_mode
