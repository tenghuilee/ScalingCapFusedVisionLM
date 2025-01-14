import enum

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_ANSWER_PLACEHOLDER = "<|answer_placeholder|>"

# Do not set enum values to 0;
# We use the logical check, value & tag == 0 or != 0
# if value == 0; 0 & any = 0; the check will not work.


class EnumTokenType(enum.Enum):
    PAD  = 0b0000000001  # the pad sequence;   1
    INST = 0b0000000010  # [INST] [/INST]; 2
    SYS  = 0b0000000100  # <<SYS>>\nxxxx\n<<SYS>>\n\n; 4
    IMG  = 0b0000001000  # image token; 8
    QUE  = 0b0000010000  # query; 16
    ANS  = 0b0000100000  # answer from gpt; 32
    BOS  = 0b0001000000  # begin of string; 64
    EOS  = 0b0010000000  # end of string; 128
    QED  = 0b0100000000  # placeholder after que; 256 a special token;
    TXT  = 0b1000000000  # general text; 512; no special meaning

    CTL = PAD | INST | SYS | BOS | EOS  # control tag

    QUE_ANS = QUE | ANS  # text query or answer
    ANS_EOS = ANS | EOS
    IMG_QED = IMG | QED
    QUE_QED = QUE | QED
    QUE_ANS_EOS = QUE | ANS | EOS
    QED_EOS = QED | EOS # qed or eos
    QED_ANS_EOS = QED | ANS | EOS # qed or answer or eos

    @staticmethod
    def is_ctl(input):
        """check if input[...] == PAD or INST or SYS or BOS or EOS"""
        return (input & EnumTokenType.CTL.value) != 0

    @staticmethod
    def not_ctl(input):
        """check if input[...] is not (PAD or INST or SYS or BOS or EOS) """
        return (input & EnumTokenType.CTL.value) == 0

    @staticmethod
    def is_the_type(input, type):
        """check if input[...] is the type"""
        return (input & type.value) != 0

    @staticmethod
    def not_the_type(input, type):
        """check if input[...] is not the type"""
        return (input & type.value) == 0
