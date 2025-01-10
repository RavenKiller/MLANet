import habitat

if habitat.__version__ == "0.1.7":
    from mlanet import (
        dagger_trainer,
        ppo_trainer_v17,
    )
else:
    from mlanet import (
        dagger_trainer,
        ppo_trainer_v21,
        real_trainer,
        alkaid_trainer,
    )

from mlanet.models import (
    # cma_policy,
    mla_policy,
    # seq2seq_policy,
    mla_ppo_policy,
)
from vlnce_baselines.common import environments
