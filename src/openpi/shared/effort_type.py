from enum import Enum, auto

class EffortType(Enum):
    """
    tokens:\n
    |<------------prefix(w=2048)----------->|<-----------suffix(w=1024)------------->|\n
    |<-images->|<-language->|<-effort(llm)->|<-effort(expert)->|<-state->|<-actions->|
    """

    NO = auto()
    """No effort will be used, but TavlaInputs still handles it for norm_stats compute."""
    STATE = auto()
    """Put current effort into last state[-14:] so it will be consider by action expert."""
    LLM = auto()
    """Project current effort into token and pass to LLM with image and language tokens.
    Projector MLP: Linear(in,2*w)->swish->Linear(2*w,w)"""
    LLM_HIS_C = auto()
    """Concat current and history effort, project into token and pass to LLM."""
    LLM_HIS_T = auto()
    """Project current and history effort into token respectively and pass to LLM."""
    EXPERT = auto()
    """Project effort into token and pass to action expert (part of LLM) with state and action tokens."""
    EXPERT_HIS_C = auto()
    """Concat current and history effort, project into token and pass to action expert."""
    EXPERT_HIS_T = auto()
    """Project current and history effort into token respectively and pass to action expert."""
    EXPERT_FUT = auto()
    """This is not an effort input type, but to predict future effort along with actions."""
    EXPERT_HIS_C_FUT = auto()
    """Input concated history effort to action expert and output future effort."""
    EXPERT_HIS_C_L_FUT = auto()
    """Input concated history effort as last token and output future effort."""

    EXPERT_HIS_C_Conv = auto()
    """Concat current and history effort, project into token via Conv1D and pass to action expert."""

    LLM_HIS_Lang = auto()
    """Concat current and history effort, convert to language description and pass to LLM."""

    LLM_HIS_FAST = auto()
    """Concat current and history effort, apply FAST token on history efforts into tokens and pass to LLM."""

    MOT = auto()