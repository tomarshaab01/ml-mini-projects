"""Grammar and spelling correction using LanguageTool NLP."""

import language_tool_python
from typing import List, Dict


def load_tool(language: str = 'en-US') -> language_tool_python.LanguageTool:
    """Load LanguageTool (downloads JRE on first run)."""
    print("[INFO] Loading LanguageTool (may take a moment on first run)...")
    return language_tool_python.LanguageTool(language)


def get_corrections(tool, text: str) -> List[Dict]:
    """Return list of detected issues with suggestions."""
    matches = tool.check(text)
    corrections = []
    for m in matches:
        corrections.append({
            'error': text[m.offset: m.offset + m.errorLength],
            'message': m.message,
            'suggestions': m.replacements[:3],
            'rule': m.ruleId
        })
    return corrections


def fix_grammar(tool, text: str) -> str:
    """Apply all corrections and return corrected text."""
    return language_tool_python.utils.correct(text, tool.check(text))


def analyze(text: str, language: str = 'en-US') -> Dict:
    """Full analysis: return raw text, corrected text, and list of corrections."""
    tool = load_tool(language)
    corrected = fix_grammar(tool, text)
    corrections = get_corrections(tool, text)
    tool.close()
    return {
        'original': text,
        'corrected': corrected,
        'num_corrections': len(corrections),
        'corrections': corrections
    }


if __name__ == '__main__':
    sample = "i went to the store yestarday and buyed some apple. their was no milk."
    result = analyze(sample)
    print(f"Original : {result['original']}")
    print(f"Corrected: {result['corrected']}")
    print(f"\nCorrections ({result['num_corrections']}):")
    for c in result['corrections']:
        print(f"  '{c['error']}' → {c['suggestions']} | {c['message']}")
