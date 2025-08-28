"""
Tests for ingenious.files.local file storage module
"""

from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from ingenious.files.local import local_FileStorageRepository


class TestLocalFileStorage:
    """Test cases for local_FileStorageRepository"""

    def setup_method(self):
        """Set up test fixtures"""
        self.mock_config = Mock()
        self.mock_fs_config = Mock()
        self.mock_fs_config.path = "/test/path"

    def test_init_with_config(self):
        """Test local_FileStorageRepository initialization"""
        storage = local_FileStorageRepository(self.mock_config, self.mock_fs_config)

        assert storage.config is self.mock_config
        assert storage.fs_config is self.mock_fs_config
        assert storage.base_path == Path("/test/path")

    @pytest.mark.asyncio
    @patch("aiofiles.open")
    @patch("pathlib.Path.mkdir")
    async def test_write_file_success(self, mock_mkdir, mock_aiofiles_open):
        """Test successful file writing"""
        storage = local_FileStorageRepository(self.mock_config, self.mock_fs_config)

        # Mock the async file context manager
        mock_file = Mock()
        mock_file.write = AsyncMock()
        mock_aiofiles_open.return_value.__aenter__ = AsyncMock(return_value=mock_file)
        mock_aiofiles_open.return_value.__aexit__ = AsyncMock(return_value=None)

        result = await storage.write_file("test content", "file.txt", "subdir")

        assert "Successfully wrote" in result
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_file.write.assert_called_once_with("test content")

    @pytest.mark.asyncio
    @patch("aiofiles.open")
    async def test_read_file_success(self, mock_aiofiles_open):
        """Test successful file reading"""
        storage = local_FileStorageRepository(self.mock_config, self.mock_fs_config)

        # Mock the async file context manager
        mock_file = Mock()
        mock_file.read = AsyncMock(return_value="file content")
        mock_aiofiles_open.return_value.__aenter__ = AsyncMock(return_value=mock_file)
        mock_aiofiles_open.return_value.__aexit__ = AsyncMock(return_value=None)

        result = await storage.read_file("file.txt", "subdir")

        assert result == "file content"
        mock_file.read.assert_called_once()

    @pytest.mark.asyncio
    @patch("aiofiles.open")
    async def test_read_file_exception(self, mock_aiofiles_open):
        """Test file reading with exception"""
        storage = local_FileStorageRepository(self.mock_config, self.mock_fs_config)

        mock_aiofiles_open.side_effect = Exception("File not found")

        result = await storage.read_file("missing.txt", "subdir")

        assert result == ""

    @pytest.mark.asyncio
    @patch("pathlib.Path.unlink")
    async def test_delete_file_success(self, mock_unlink):
        """Test successful file deletion"""
        storage = local_FileStorageRepository(self.mock_config, self.mock_fs_config)

        result = await storage.delete_file("file.txt", "subdir")

        assert "Successfully deleted" in result
        mock_unlink.assert_called_once()

    @pytest.mark.asyncio
    @patch("pathlib.Path.unlink")
    async def test_delete_file_exception(self, mock_unlink):
        """Test file deletion with exception"""
        storage = local_FileStorageRepository(self.mock_config, self.mock_fs_config)

        mock_unlink.side_effect = Exception("Delete failed")

        result = await storage.delete_file("file.txt", "subdir")

        assert "Failed to delete" in result

    @pytest.mark.asyncio
    @patch("pathlib.Path.iterdir")
    async def test_list_files_success(self, mock_iterdir):
        """Test successful file listing"""
        storage = local_FileStorageRepository(self.mock_config, self.mock_fs_config)

        # Mock file objects
        mock_file1 = Mock()
        mock_file1.name = "file1.txt"
        mock_file1.is_file.return_value = True

        mock_file2 = Mock()
        mock_file2.name = "file2.txt"
        mock_file2.is_file.return_value = True

        mock_dir = Mock()
        mock_dir.name = "subdir"
        mock_dir.is_file.return_value = False

        mock_iterdir.return_value = [mock_file1, mock_file2, mock_dir]

        result = await storage.list_files("testdir")

        # Should return string representation of file list
        assert "file1.txt" in result
        assert "file2.txt" in result
        # Should not include directories
        assert "subdir" not in result

    @pytest.mark.asyncio
    @patch("pathlib.Path.iterdir")
    async def test_list_files_exception(self, mock_iterdir):
        """Test file listing with exception"""
        storage = local_FileStorageRepository(self.mock_config, self.mock_fs_config)

        mock_iterdir.side_effect = Exception("Directory not found")

        result = await storage.list_files("missing_dir")

        assert "Failed to list files" in result

    @pytest.mark.asyncio
    @patch("pathlib.Path.exists")
    async def test_check_if_file_exists_true(self, mock_exists):
        """Test checking if file exists - exists"""
        storage = local_FileStorageRepository(self.mock_config, self.mock_fs_config)

        mock_exists.return_value = True

        result = await storage.check_if_file_exists("subdir", "file.txt")

        assert result is True
        mock_exists.assert_called_once()

    @pytest.mark.asyncio
    @patch("pathlib.Path.exists")
    async def test_check_if_file_exists_false(self, mock_exists):
        """Test checking if file exists - doesn't exist"""
        storage = local_FileStorageRepository(self.mock_config, self.mock_fs_config)

        mock_exists.return_value = False

        result = await storage.check_if_file_exists("subdir", "missing.txt")

        assert result is False

    @pytest.mark.asyncio
    @patch("pathlib.Path.exists")
    async def test_check_if_file_exists_exception(self, mock_exists):
        """Test checking if file exists with exception"""
        storage = local_FileStorageRepository(self.mock_config, self.mock_fs_config)

        mock_exists.side_effect = Exception("Path error")

        result = await storage.check_if_file_exists("subdir", "file.txt")

        assert result is False

    @pytest.mark.asyncio
    async def test_get_base_path(self):
        """Test getting base path"""
        storage = local_FileStorageRepository(self.mock_config, self.mock_fs_config)

        result = await storage.get_base_path()

        assert result == "/test/path"

    def test_inherits_from_interface(self):
        """Test that class inherits from IFileStorage"""
        from ingenious.files.files_repository import IFileStorage

        assert issubclass(local_FileStorageRepository, IFileStorage)

    def test_path_construction(self):
        """Test that paths are constructed correctly"""
        storage = local_FileStorageRepository(self.mock_config, self.mock_fs_config)

        # Test that the base path is set correctly
        assert storage.base_path == Path("/test/path")

        # Test path composition works
        test_path = storage.base_path / "subdir" / "file.txt"
        expected = Path("/test/path/subdir/file.txt")
        assert test_path == expected
