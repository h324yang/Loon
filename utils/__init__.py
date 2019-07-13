from utils.utils import indicator_vector
from utils.utils import truncate_seq_pair
from utils.utils import get_span_labels


never_split = ("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]",
              '[ORGANIZATION-SUBJ]', '[PERSON-OBJ]', '[PERSON-SUBJ]',
              '[ORGANIZATION-OBJ]', '[NUMBER-OBJ]', '[DATE-OBJ]',
              '[NATIONALITY-OBJ]', '[LOCATION-OBJ]', '[TITLE-OBJ]',
              '[CITY-OBJ]', '[MISC-OBJ]', '[COUNTRY-OBJ]',
              '[CRIMINAL_CHARGE-OBJ]', '[RELIGION-OBJ]',
              '[DURATION-OBJ]', '[URL-OBJ]', '[STATE_OR_PROVINCE-OBJ]',
              '[IDEOLOGY-OBJ]', '[CAUSE_OF_DEATH-OBJ]')



