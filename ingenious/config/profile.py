import json
import os
import re
from pathlib import Path
from typing import Any, Optional

import yaml
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from pydantic import ValidationError

from ingenious.models import profile as profile_models


def substitute_environment_variables(yaml_content: str) -> str:
    """
    Substitute environment variables in YAML content.
    Supports patterns like ${VAR_NAME} and ${VAR_NAME:default_value}
    """

    def replacer(match: re.Match[str]) -> str:
        var_expr = match.group(1)
        if ":" in var_expr:
            var_name, default_value = var_expr.split(":", 1)
            env_value = os.getenv(var_name)
            if env_value is None:
                # Provide more helpful guidance for missing environment variables
                if var_name in ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_BASE_URL"]:
                    print(
                        f"⚠️  Required environment variable {var_name} not found. "
                        f"Using default value '{default_value}'. "
                        f"Please set {var_name} in your .env file for proper operation."
                    )
                elif "placeholder" in default_value.lower():
                    print(
                        f"ℹ️  Optional service variable {var_name} not configured. "
                        f"Using placeholder '{default_value}'. This is normal for minimal setups."
                    )
                return default_value
            return env_value
        else:
            var_name = var_expr
            env_value = os.getenv(var_name)
            if env_value is None:
                print(
                    f"❌ Critical: Environment variable {var_name} not found and no default provided. "
                    f"Please set {var_name} in your .env file or provide a default value in profiles.yml"
                )
                return match.group(0)  # Return original if no env var found
            return env_value

    # Pattern matches ${VAR_NAME} or ${VAR_NAME:default}
    pattern = r"\$\{([^}]+)\}"
    return re.sub(pattern, replacer, yaml_content)


class Profiles:
    def __init__(self, profiles_path: Optional[str] = None) -> None:
        self.profiles: profile_models.Profiles = Profiles._get_profiles(profiles_path)

    @staticmethod
    def from_yaml_str(profile_yml: str) -> Any:
        # Substitute environment variables in the profile YAML
        profile_yml = substitute_environment_variables(profile_yml)
        yaml_data = yaml.safe_load(profile_yml)
        json_data = json.dumps(yaml_data)
        try:
            profiles = profile_models.Profiles.model_validate_json(json_data).root
        except ValidationError as e:
            print("\n" + "=" * 80)
            print("❌ PROFILE VALIDATION FAILED")
            print("=" * 80)
            print("🔧 Configuration issues found. Here's how to fix them:\n")

            error_count = len(e.errors())
            for i, error in enumerate(e.errors(), 1):
                field_path = " -> ".join(str(x) for x in error["loc"])
                error_msg = error["msg"]

                print(f"� Error {i}/{error_count}: {field_path}")
                print(f"   Issue: {error_msg}")

                # Provide specific, actionable solutions based on common errors
                if "string_type" in error_msg and field_path.endswith("api_key"):
                    print("   💡 Solution:")
                    print(
                        "      1. Make sure AZURE_OPENAI_API_KEY is set in your .env file"
                    )
                    print(
                        "      2. Example: AZURE_OPENAI_API_KEY=your-actual-api-key-here"
                    )
                    print(
                        "      3. Check your Azure OpenAI resource for the correct key"
                    )
                elif "string_type" in error_msg and field_path.endswith("base_url"):
                    print("   💡 Solution:")
                    print(
                        "      1. Make sure AZURE_OPENAI_BASE_URL is set in your .env file"
                    )
                    print(
                        "      2. Example: AZURE_OPENAI_BASE_URL=https://your-resource.cognitiveservices.azure.com/"
                    )
                    print(
                        "      3. Find this URL in your Azure OpenAI resource overview"
                    )
                elif "string_type" in error_msg and "password" in field_path:
                    print("   💡 Solution:")
                    print(
                        "      1. Set WEB_AUTH_PASSWORD='' in .env to disable authentication"
                    )
                    print(
                        "      2. Or set a strong password: WEB_AUTH_PASSWORD=your-secure-password"
                    )
                    print(
                        "      3. Authentication can be disabled by setting enable: false"
                    )
                elif "string_type" in error_msg and any(
                    x in field_path for x in ["azure_search", "azure_sql", "storage"]
                ):
                    print("   💡 Solution:")
                    print(
                        "      1. For minimal setup, these can remain as placeholder values"
                    )
                    print(
                        "      2. These services are optional for basic bike-insights workflow"
                    )
                    print(
                        "      3. Copy the minimal template: cp ingenious_extensions/profiles.minimal.yml ./profiles.yml"
                    )
                elif "required" in error_msg.lower():
                    print("   💡 Solution:")
                    print("      1. This field is required and cannot be empty")
                    print(
                        "      2. Check your .env file for the corresponding environment variable"
                    )
                    print(
                        "      3. Refer to .env.example for the complete list of variables"
                    )
                else:
                    print("   💡 Solution:")
                    print("      1. Check your profiles.yml syntax and .env variables")
                    print(
                        "      2. Compare with working examples in ingenious_extensions/"
                    )
                    print(
                        "      3. Run 'ingen validate' for detailed configuration checks"
                    )
                print()

            print("📋 QUICK FIX COMMANDS:")
            print("   # Use minimal working template:")
            print("   cp ingenious_extensions/profiles.minimal.yml ./profiles.yml")
            print()
            print("   # Check your environment variables:")
            print("   cat .env | grep AZURE_OPENAI")
            print()
            print("   # Validate configuration:")
            print("   uv run ingen validate")
            print()
            print("📖 HELPFUL RESOURCES:")
            print("   • Documentation: see docs/troubleshooting/README.md")
            print("   • Quick Start: uv run ingen workflows bike-insights")
            print("   • Examples: check ingenious_extensions/ directory")
            print("=" * 80)
            raise e
        except Exception as e:
            print(f"\n❌ Unexpected error during profile validation: {e}")
            print(
                "💡 This might be a YAML syntax error. Check your profiles.yml file formatting."
            )
            raise e
        return profiles

    @staticmethod
    def from_yaml(file_path: str) -> Any:
        with open(file_path, "r") as file:
            file_str = file.read()
            # Substitute environment variables before parsing
            file_str = substitute_environment_variables(file_str)
            profiles = Profiles.from_yaml_str(file_str)
            return profiles

    @staticmethod
    def _get_profiles(profiles_path: Optional[str] = None) -> Any:
        # Check if os.getenv('INGENIOUS_PROFILE') is set
        if os.getenv("APPSETTING_INGENIOUS_PROFILE", "") != "":
            # print("Profile JSON loaded from environment variable")
            profile_string = os.getenv("APPSETTING_INGENIOUS_PROFILE", "{}")
            profile_object = json.loads(profile_string)
            # Convert the json string to a yaml string
            profile_yml = yaml.dump(profile_object)
            profile = Profiles.from_yaml_str(profile_yml)
            return profile

        # Load the configuration from the YAML file
        if profiles_path is None or profiles_path == "":
            if os.getenv("INGENIOUS_PROFILE_PATH", "") != "":
                print("Profile Path loaded from environment variable")
                profiles_path_object = Path(os.getenv("INGENIOUS_PROFILE_PATH", ""))
            else:
                print("Profile loaded from default path")
                home_directory = os.path.expanduser("~")
                profiles_path_object = (
                    Path(home_directory) / Path(".ingenious") / Path("profiles.yml")
                )
        else:
            profiles_path_object = Path(profiles_path)

        if profiles_path_object.is_file():
            print("Profile loaded from file")
            profiles = Profiles.from_yaml(file_path=str(profiles_path_object))
        else:
            print(f"Profile not found at {profiles_path}")
            print("Trying to load profile from key vault")
            profiles_yml = get_kv_secret(secretName="profile")
            profiles = Profiles.from_yaml_str(profiles_yml)

        return profiles


def get_kv_secret(secretName: str) -> str:
    try:
        keyVaultName = os.environ["KEY_VAULT_NAME"]
    except KeyError:
        raise ValueError("KEY_VAULT_NAME environment variable not set")

    KVUri = f"https://{keyVaultName}.vault.azure.net"
    credential = DefaultAzureCredential()
    client = SecretClient(vault_url=KVUri, credential=credential)
    secret = client.get_secret(secretName)
    return secret.value or ""
