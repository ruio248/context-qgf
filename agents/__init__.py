from agents.bc import BCAgent
from agents.cfgrl import CFGRLAgent
from agents.context_qgf import ContextQGFAgent
from agents.context_qgf_adapter import ContextQGFAdapterAgent
from agents.dcgql import DCGQLAgent
from agents.dsrl import DSRLAgent
from agents.edp import EDPAgent
from agents.fawac import FAWACAgent
from agents.fbrac import FBRACAgent
from agents.fql import FQLAgent
from agents.grad_step import GradStepAgent
from agents.ifql import IFQLAgent
from agents.iql import IQLAgent
from agents.iql_diffusion import IQLDiffusionAgent
from agents.qam import QAMAgent
from agents.qgf import QGFAgent
from agents.qgf_qv_finetune import QGFQVFinetuneAgent
from agents.robust_q import RobustQAgent
from agents.sac import SACAgent

agents = dict(
    bc=BCAgent,
    fql=FQLAgent,
    ifql=IFQLAgent,
    iql=IQLAgent,
    iql_diffusion=IQLDiffusionAgent,
    cfgrl=CFGRLAgent,
    context_qgf=ContextQGFAgent,
    context_qgf_adapter=ContextQGFAdapterAgent,
    qgf=QGFAgent,
    qgf_qv_finetune=QGFQVFinetuneAgent,
    robust_q=RobustQAgent,
    sac=SACAgent,
    qam=QAMAgent,
    edp=EDPAgent,
    dcgql=DCGQLAgent,
    dsrl=DSRLAgent,
    fawac=FAWACAgent,
    fbrac=FBRACAgent,
    grad_step=GradStepAgent,
)
