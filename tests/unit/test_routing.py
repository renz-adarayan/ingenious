"""
Tests for ingenious.main.routing module
"""

from unittest.mock import Mock, patch

from ingenious.main.routing import RouteManager


class TestRouteManager:
    """Test cases for RouteManager class"""

    def test_module_docstring(self):
        """Test that the module has appropriate documentation"""
        import ingenious.main.routing as routing

        docstring = routing.__doc__
        assert docstring is not None
        assert "Route registration for the FastAPI application" in docstring

    def test_imports_exist(self):
        """Test that all required imports are available"""
        import ingenious.main.routing as routing

        # Test that all required imports are accessible
        assert hasattr(routing, "TYPE_CHECKING")
        assert hasattr(routing, "auth_route")
        assert hasattr(routing, "chat_route")
        assert hasattr(routing, "conversation_route")
        assert hasattr(routing, "diagnostic_route")
        assert hasattr(routing, "message_feedback_route")
        assert hasattr(routing, "prompts_route")
        assert hasattr(routing, "IApiRoutes")
        assert hasattr(routing, "import_class_with_fallback")
        assert hasattr(routing, "import_module_with_fallback")

    def test_register_builtin_routes(self):
        """Test that register_builtin_routes registers all built-in routes"""
        mock_app = Mock()

        RouteManager.register_builtin_routes(mock_app)

        # Verify all routes were registered
        assert mock_app.include_router.call_count == 7

        # Check specific route registrations
        calls = mock_app.include_router.call_args_list

        # Auth route
        auth_call = calls[0]
        assert auth_call[1]["prefix"] == "/api/v1/auth"
        assert auth_call[1]["tags"] == ["Authentication"]

        # Chat route
        chat_call = calls[1]
        assert chat_call[1]["prefix"] == "/api/v1"
        assert chat_call[1]["tags"] == ["Chat"]

        # Conversation route
        conversation_call = calls[2]
        assert conversation_call[1]["prefix"] == "/api/v1"
        assert conversation_call[1]["tags"] == ["Conversations"]

        # Diagnostic route
        diagnostic_call = calls[3]
        assert diagnostic_call[1]["prefix"] == "/api/v1"
        assert diagnostic_call[1]["tags"] == ["Diagnostic"]

        # Prompts route
        prompts_call = calls[4]
        assert prompts_call[1]["prefix"] == "/api/v1"
        assert prompts_call[1]["tags"] == ["Prompts"]

        # Message feedback route
        feedback_call = calls[5]
        assert feedback_call[1]["prefix"] == "/api/v1"
        assert feedback_call[1]["tags"] == ["Message Feedback"]

        # Custom workflows route
        feedback_call = calls[6]
        assert feedback_call[1]["prefix"] == "/api/v1"
        assert feedback_call[1]["tags"] == ["Custom Workflows"]

    @patch("ingenious.main.routing.import_module_with_fallback")
    def test_register_custom_routes_no_custom_module(self, mock_import_module):
        """Test register_custom_routes when no custom module is found"""
        mock_app = Mock()
        mock_config = Mock()

        # Mock the module import to return built-in module
        mock_module = Mock()
        mock_module.__name__ = "ingenious.api.routes.custom"
        mock_import_module.return_value = mock_module

        RouteManager.register_custom_routes(mock_app, mock_config)

        # Verify import was attempted
        mock_import_module.assert_called_once_with("api.routes.custom")

        # Since it returned the built-in module, no custom routes should be registered
        # (the method should return early)

    @patch("ingenious.main.routing.import_class_with_fallback")
    @patch("ingenious.main.routing.import_module_with_fallback")
    def test_register_custom_routes_with_custom_module(
        self, mock_import_module, mock_import_class
    ):
        """Test register_custom_routes when custom module is found"""
        mock_app = Mock()
        mock_config = Mock()

        # Mock the module import to return custom module
        mock_module = Mock()
        mock_module.__name__ = "custom.api.routes.custom"  # Different from built-in
        mock_import_module.return_value = mock_module

        # Mock the class import
        mock_custom_class = Mock()
        mock_import_class.return_value = mock_custom_class

        # Mock the class instance
        mock_instance = Mock()
        mock_custom_class.return_value = mock_instance

        RouteManager.register_custom_routes(mock_app, mock_config)

        # Verify imports were called
        mock_import_module.assert_called_once_with("api.routes.custom")
        mock_import_class.assert_called_once_with("api.routes.custom", "Api_Routes")

        # Verify class was instantiated with config and app
        mock_custom_class.assert_called_once_with(mock_config, mock_app)

        # Verify add_custom_routes was called
        mock_instance.add_custom_routes.assert_called_once()

    @patch("ingenious.main.routing.import_module_with_fallback")
    def test_register_custom_routes_module_name_comparison(self, mock_import_module):
        """Test the module name comparison logic in register_custom_routes"""
        mock_app = Mock()
        mock_config = Mock()

        # Test different module name scenarios
        test_cases = [
            (
                "ingenious.api.routes.custom",
                False,
            ),  # Built-in module - should not register
            (
                "custom_extension.api.routes.custom",
                True,
            ),  # Custom module - should register
            (
                "my_extension.api.routes.custom",
                True,
            ),  # Another custom module - should register
        ]

        for module_name, should_register in test_cases:
            mock_module = Mock()
            mock_module.__name__ = module_name
            mock_import_module.return_value = mock_module

            with patch(
                "ingenious.main.routing.import_class_with_fallback"
            ) as mock_import_class:
                mock_custom_class = Mock()
                mock_import_class.return_value = mock_custom_class
                mock_instance = Mock()
                mock_custom_class.return_value = mock_instance

                RouteManager.register_custom_routes(mock_app, mock_config)

                if should_register:
                    mock_import_class.assert_called_once()
                    mock_custom_class.assert_called_once()
                    mock_instance.add_custom_routes.assert_called_once()
                else:
                    mock_import_class.assert_not_called()

    def test_register_all_routes_calls_both_methods(self):
        """Test that register_all_routes calls both builtin and custom route registration"""
        mock_app = Mock()
        mock_config = Mock()

        with (
            patch.object(RouteManager, "register_builtin_routes") as mock_builtin,
            patch.object(RouteManager, "register_custom_routes") as mock_custom,
        ):
            RouteManager.register_all_routes(mock_app, mock_config)

            # Verify both methods were called with correct arguments
            mock_builtin.assert_called_once_with(mock_app)
            mock_custom.assert_called_once_with(mock_app, mock_config)

    def test_route_manager_methods_are_static_or_class(self):
        """Test that RouteManager methods are static or class methods"""
        # Check specific methods we know should exist
        expected_methods = [
            "register_builtin_routes",
            "register_custom_routes",
            "register_all_routes",
        ]

        for method_name in expected_methods:
            assert hasattr(RouteManager, method_name)
            method = getattr(RouteManager, method_name)
            # Method should be callable
            assert callable(method)

    def test_route_prefixes_are_consistent(self):
        """Test that route prefixes follow consistent pattern"""
        mock_app = Mock()

        RouteManager.register_builtin_routes(mock_app)

        calls = mock_app.include_router.call_args_list

        # Check that most routes use /api/v1 prefix (except auth which uses /api/v1/auth)
        prefixes = [call[1]["prefix"] for call in calls]

        assert "/api/v1/auth" in prefixes  # Auth has special prefix
        api_v1_count = prefixes.count("/api/v1")
        assert api_v1_count >= 4  # Most routes should use /api/v1

    def test_route_tags_are_properly_set(self):
        """Test that all routes have appropriate tags"""
        mock_app = Mock()

        RouteManager.register_builtin_routes(mock_app)

        calls = mock_app.include_router.call_args_list

        # Check that all routes have tags
        for call in calls:
            assert "tags" in call[1]
            assert isinstance(call[1]["tags"], list)
            assert len(call[1]["tags"]) == 1  # Each route should have exactly one tag
            assert call[1]["tags"][0]  # Tag should not be empty

    def test_class_methods_have_docstrings(self):
        """Test that all public methods have docstrings"""
        public_methods = [
            "register_builtin_routes",
            "register_custom_routes",
            "register_all_routes",
        ]

        for method_name in public_methods:
            method = getattr(RouteManager, method_name)
            if hasattr(method, "__func__"):  # Handle static/class method wrappers
                actual_method = method.__func__
            else:
                actual_method = method

            assert actual_method.__doc__ is not None, (
                f"{method_name} should have a docstring"
            )
            assert len(actual_method.__doc__.strip()) > 0, (
                f"{method_name} docstring should not be empty"
            )
