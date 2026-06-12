import sys
from graph import build_workflow, AgentState

def test_compilation():
    print("Testing LangGraph compilation...")
    try:
        app = build_workflow()
        print("[OK] LangGraph Compiled successfully!")
        
        # Verify state schema contains expected fields
        expected_keys = [
            "resume_text", "job_description", "parsed_resume", 
            "skill_analysis", "job_match", "interview_prep",
            "api_key", "api_provider", "model_name", "errors"
        ]
        
        print("Verifying state keys...")
        # Get schema keys from TypedDict annotations
        state_keys = list(AgentState.__annotations__.keys())
        for key in expected_keys:
            if key not in state_keys:
                print(f"[FAIL] State missing key: {key}")
                sys.exit(1)
            print(f"  - Verified key: {key}")
            
        print("[OK] State schema verification complete!")
        print("All tests passed.")
        
    except Exception as e:
        print(f"[ERROR] Compilation failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    test_compilation()
