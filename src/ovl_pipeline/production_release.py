"""Closed final release inventories and explicit operator-attestation semantics.

A publisher signature on this inventory does not execute reconstruction or replay.
A consumer must separately choose identity/download checks or actual full replay.
"""
from dataclasses import replace
from pathlib import Path
import re
import shutil

from .anchoring import PublisherPolicy,WORKFLOW,verify_anchor
from .canonical import EvidenceError,confined,digest,file_hash,inventory,read_json,require_digest,verify_inventory,write_json
from .production_contract import validate as registration_contract
from .production_export import FILES,load_model
from .schema import fields,integer

RELEASE_WORKFLOW='.github/workflows/release-models.yml'
PAYLOAD=sorted(['LICENSE','MODEL_SOURCES.md','README.md','demonstrations.json','registration.json',*['model/'+n for n in FILES]])
EVIDENCE=sorted(['context.json','verification.json','reconstruction.json','replay.json','evaluation.json','exports.json'])
CHECKS={'publisher_ancestry':'PASS','complete_raw_reconstruction':'PASS','continuous_complete_replay':'PASS',
        'both_export_mappings':'PASS','complete_heldout_evaluation':'PASS','fixed_greedy_inference':'PASS'}


class ReleasePublisherPolicy(PublisherPolicy):
    def validate(self):
        if self.workflow!=RELEASE_WORKFLOW:raise EvidenceError('wrong final release workflow identity')
        PublisherPolicy.validate(replace(self,workflow=WORKFLOW))


def publication_id(value):
    if type(value) is not str or not re.fullmatch(r'release-[1-9][0-9]{0,5}',value):raise EvidenceError('explicit bounded publication identity required')
    return value


def model_repo(r,phase,publication='release-1'):
    if phase not in ('base','chat'):raise EvidenceError('explicit model phase required')
    registration_contract(r)
    return 'AOSSIE/openverifiable-'+r['run_id']+'-'+r['attempt_id']+'-'+publication_id(publication)+'-'+phase


def entries(value,names,maximum):
    if type(value) is not list or len(value)!=len(names):raise EvidenceError('complete closed release inventory required')
    for e in value:
        fields(e,'path bytes sha256','release inventory member');require_digest(e['sha256'])
        integer(e['bytes'],1,maximum,'release file length')
    if [e['path'] for e in value]!=names or sum(e['bytes'] for e in value)>maximum:
        raise EvidenceError('release inventory names or total size differ')


def archive_shape(value):
    fields(value,'repo revision prefix inventory','release evidence archive')
    if (type(value['repo']) is not str or not re.fullmatch(r'AOSSIE/openverifiable-[a-z0-9-]+-evidence',value['repo'])
        or type(value['revision']) is not str or not re.fullmatch('[0-9a-f]{40}',value['revision'])
        or type(value['prefix']) is not str or not re.fullmatch(r'release-evidence/[0-9a-f]{64}',value['prefix'])):
        raise EvidenceError('immutable approved release evidence archive required')
    entries(value['inventory'],EVIDENCE,64*1024**2)
    if value['prefix']!='release-evidence/'+digest(value['inventory']):raise EvidenceError('release evidence prefix differs from inventory root')


def validate(value):
    fields(value,'schema scope publication_id registration_sha256 source_statement_sha256 preparation_sha256 code models evidence report_roots claims','final release')
    if value['schema']!='ovl.production-release.v1' or value['scope']!='operator-attestation-of-complete-reconstruction-and-replay':
        raise EvidenceError('unsupported final release schema/scope')
    publication_id(value['publication_id'])
    for n in ('registration_sha256','source_statement_sha256','preparation_sha256'):require_digest(value[n])
    code=value['code'];fields(code,'repository revision code_root loader','release code')
    if (code['repository']!='AOSSIE-Org/OpenVerifiableLLM' or not re.fullmatch('[0-9a-f]{40}',code['revision'])
        or code['loader']!='ovl_pipeline.production_export'):
        raise EvidenceError('release loader source is not the approved pinned repository')
    require_digest(code['code_root']);archive_shape(value['evidence'])
    fields(value['models'],'base chat','both released models')
    for phase,model in value['models'].items():
        fields(model,'repo phase model_root files','released model')
        if (model['phase']!=phase or type(model['repo']) is not str
            or not re.fullmatch(r'AOSSIE/openverifiable-[a-z0-9-]+-'+value['publication_id']+'-'+phase,model['repo'])):
            raise EvidenceError('new descriptive AOSSIE model repository required')
        require_digest(model['model_root']);entries(model['files'],PAYLOAD,768*1024**2)
        selected=next(e for e in model['files'] if e['path']=='registration.json')
        if selected['sha256']!=value['registration_sha256']:raise EvidenceError('model payload registration differs')
    fields(value['report_roots'],'verification reconstruction replay evaluation exports','release reports')
    recorded={e['path']:e['sha256'] for e in value['evidence']['inventory']}
    for name,root in value['report_roots'].items():
        require_digest(root)
        if recorded[name+'.json']!=root:raise EvidenceError('release report/archive parent differs')
    if value['claims']!={'training_verified_by':'project-operator','independent_third_party':False,
                         'signature_establishes':'publisher-endorsement-and-integrity-only',
                         'consumer_training_recomputed':False,'factual_accuracy':'NOT_ESTABLISHED_BY_PROVENANCE_OR_TOKEN_LOSS'}:
        raise EvidenceError('unsupported or overstated release claim')
    return value


def reports(r,directory):
    """Check assertion relationships only; this function performs no training."""
    from .production_chain import verify_chain
    from .production_verify import STAGES
    values={n:read_json(confined(directory,n+'.json')) for n in ('context','verification','reconstruction','replay','evaluation','exports')}
    c=values['context'];fields(c,'schema closing_request final_progress','release navigation')
    if c['schema']!='ovl.release-context.v1':raise EvidenceError('unsupported release navigation')
    from .progress_commitment import validate_request,archive_shape as progress_archive_shape
    validate_request(c['closing_request'])
    fields(c['final_progress'],'archive policy','final progress selection')
    root=digest(r);envelopes=c['closing_request']['envelopes'];verify_chain(r,root,envelopes,complete=True)
    if c['closing_request']['registration_request']['registration_sha256']!=root:
        raise EvidenceError('release navigation selects a different registration')
    progress_archive_shape(c['final_progress']['archive'],f'production-progress/{root}/progress-{len(envelopes)-1:05d}',
                           ['statement.json','statement.sigstore.json'],4*1024**2)
    from .progress_anchoring import ProgressPublisherPolicy
    ProgressPublisherPolicy(**c['final_progress']['policy']).validate()
    reconstruction=values['reconstruction'];replay=values['replay'];verification=values['verification'];evaluation=values['evaluation'];exports=values['exports']
    fields(reconstruction,'result scope preparation_sha256 source_commitment_sha256 full_reconstruction_compared execution_observation_sha256 stages_executed_this_run stages_adopted_from_local_cache training_replay production_training_admission','reconstruction report')
    composed=verification.get('schema')=='ovl.complete-computation-verification.v2'
    fields(verification,'schema result registration_sha256 scope checks endorsements reconstruction reconstructed_artifacts replay_report_sha256 replay_process exports evaluation_sha256 raw_inputs base_model_root chat_model_root reconstruction_and_input_validation_ms total_ms performed_by attested_by locally_recomputed independent_third_party public_release_download_verification factual_accuracy'+(' execution' if composed else ''),'complete computation report')
    fields(replay,'schema result scope registration_sha256 session_sha256 chain_sha256 endorsements artifact_check comparisons recovery_checkpoints updates_recomputed targets_recomputed initial_state_regenerated prover_checkpoints_restored base_model_root chat_model_root setup_ms numerical_replay_ms performed_by independent_third_party raw_transformation_reconstruction public_download_verification cost_guard_admission end_to_end_release_verification','numerical replay report')
    fields(evaluation,'schema registration_sha256 split stream_sha256 subset_sampling recipe runtime models factual_accuracy independent_third_party','held-out evaluation report')
    fields(exports,'schema status registration_sha256 chain_sha256 exports endorsements prepared_artifact_check full_raw_reconstruction continuous_numerical_replay public_download_verification factual_accuracy','candidate export report')
    if (reconstruction.get('result')!='PASS' or reconstruction.get('scope')!='complete-source-preparation'
        or reconstruction.get('full_reconstruction_compared') is not True
        or reconstruction.get('preparation_sha256')!=r['preparation_sha256']
        or reconstruction.get('source_commitment_sha256')!=r['source_statement_sha256']
        or reconstruction.get('stages_executed_this_run')!=STAGES or reconstruction.get('stages_adopted_from_local_cache')!=[]):
        raise EvidenceError('release lacks complete fresh reconstruction assertion')
    if (verification.get('schema') not in ('ovl.complete-computation-verification.v1','ovl.complete-computation-verification.v2') or verification.get('result')!='PASS'
        or verification['scope']!='complete-raw-transformations-fresh-initialization-all-updates-both-exported-models'
        or verification.get('registration_sha256')!=root or verification.get('checks')!=CHECKS
        or verification.get('attested_by') is not None or verification.get('public_release_download_verification')!='NOT_RUN'
        or verification.get('locally_recomputed') is not (not composed) or verification.get('independent_third_party') is not False
        or verification.get('reconstruction')!=reconstruction or verification.get('replay_report_sha256')!=digest(replay)
        or verification.get('evaluation_sha256')!=digest(evaluation)):
        raise EvidenceError('complete computation assertion missing or disconnected')
    if composed:
        from .production_composition import validate as validate_composition
        validate_composition(verification['execution'],r,reconstruction,replay,verification['replay_process'])
        if (verification['performed_by']!='project-operator-separate-recorded-executions'
            or verification['execution']['assembly']['raw_inventory_sha256']!=verification['raw_inputs']['complete_inventory_sha256']):
            raise EvidenceError('composed computation scope or raw inventory differs')
    if (replay.get('schema')!='ovl.production-numerical-replay.v1' or replay.get('result')!='PASS'
        or replay['scope']!='fresh-regenerated-initialization-continuous-two-phase-all-update-state-comparison'
        or replay['independent_third_party'] is not False
        or replay.get('registration_sha256')!=root or replay.get('initial_state_regenerated') is not True
        or replay.get('prover_checkpoints_restored') is not False or replay.get('chain_sha256')!=digest(envelopes)
        or replay.get('updates_recomputed')!={p:r['coverage'][p]['updates'] for p in ('wikipedia','conversation')}
        or replay.get('targets_recomputed')!={p:r['coverage'][p]['targets'] for p in ('wikipedia','conversation')}
        or len(replay.get('comparisons',[]))!=len(envelopes)):
        raise EvidenceError('release lacks continuous complete replay assertion')
    for counts in ('updates_recomputed','targets_recomputed'):
        for phase in ('wikipedia','conversation'):integer(replay[counts][phase],1,2**53-1,'nonempty replay coverage')
    for env,comparison in zip(envelopes,replay['comparisons']):
        body=env['body']
        if (comparison['index']!=body['index'] or comparison['boundary_sha256']!=digest(env)
            or comparison['state_root']!=body['checkpoint']['state_root'] or comparison['control']!=body['control'] or comparison['result']!='PASS'):
            raise EvidenceError('replay comparison ancestry differs from complete public chain')
    if (evaluation.get('schema')!='ovl.complete-heldout-evaluation.v1' or evaluation.get('registration_sha256')!=root
        or evaluation.get('subset_sampling') is not False or evaluation.get('split')!='official-conversation-validation'
        or evaluation.get('factual_accuracy')!='NOT_ESTABLISHED_BY_PROVENANCE_OR_TOKEN_LOSS'):
        raise EvidenceError('held-out evaluation assertion missing or overstated')
    if (exports.get('schema')!='ovl.production-candidate-export.v1' or exports.get('registration_sha256')!=root
        or exports.get('status')!='EXPORTED_NOT_TRAINING_VERIFIED'
        or exports.get('chain_sha256')!=digest(envelopes) or set(exports.get('exports',{}))!={'base','chat'}):
        raise EvidenceError('export evidence does not bind both selected phase states')
    for phase in ('base','chat'):
        model_root=replay[phase+'_model_root'];require_digest(model_root)
        entries(exports['exports'][phase]['files'],FILES,768*1024**2)
        integer(evaluation['models'][phase]['targets'],1,2**53-1,'held-out evaluation targets')
        if (verification[phase+'_model_root']!=model_root or exports['exports'][phase]['model_root']!=model_root
            or evaluation['models'][phase]['model_root']!=model_root):raise EvidenceError('model roots disagree across release reports')
        boundary=next(e for e in envelopes if e['body']['kind']==('base' if phase=='base' else 'final'))
        if (exports['exports'][phase]['boundary_sha256']!=digest(boundary)
            or exports['exports'][phase]['checkpoint_state_root']!=boundary['body']['checkpoint']['state_root']):
            raise EvidenceError('export boundary differs from selected public phase endpoint')
    return values


def build(r,report_directory,evidence_archive,payloads,*,publication='release-1'):
    values=reports(r,report_directory);archive_shape(evidence_archive);verify_inventory(report_directory,evidence_archive['inventory'])
    models={}
    for phase in ('base','chat'):
        directory=payloads[phase];root=values['replay'][phase+'_model_root']
        if read_json(directory/'registration.json')!=r:raise EvidenceError('release payload registration differs')
        load_model(directory/'model',r,phase,expected_model_root=root)
        prepared=read_json(directory/'model/preparation.json')
        expected_validation=prepared['streams']['conversation-validation']
        if (values['evaluation']['stream_sha256']!=digest(expected_validation)
            or values['evaluation']['models'][phase]['targets']!=expected_validation['targets']):
            raise EvidenceError('evaluation assertion does not cover the registered full held-out stream')
        selected=inventory(directory,PAYLOAD)
        # Additional payloads may hide unsafe binaries or private material.
        actual=[]
        for p in directory.rglob('*'):
            if p.is_symlink() or not(p.is_dir() or p.is_file()):raise EvidenceError('nonregular release payload')
            if p.is_file():actual.append(p.relative_to(directory).as_posix())
        if sorted(actual)!=PAYLOAD:raise EvidenceError('unregistered release payload files')
        if read_json(directory/'demonstrations.json')!=values['evaluation']['models'][phase]['demonstrations']:
            raise EvidenceError('release demonstrations differ from evaluated model')
        models[phase]={'repo':model_repo(r,phase,publication),'phase':phase,'model_root':root,'files':selected}
    result={'schema':'ovl.production-release.v1','publication_id':publication_id(publication),'scope':'operator-attestation-of-complete-reconstruction-and-replay',
            'registration_sha256':digest(r),'source_statement_sha256':r['source_statement_sha256'],'preparation_sha256':r['preparation_sha256'],
            'code':{'repository':'AOSSIE-Org/OpenVerifiableLLM','revision':r['code_revision'],'code_root':r['code_root'],
                    'loader':'ovl_pipeline.production_export'},'models':models,'evidence':evidence_archive,
            'report_roots':{n:digest(values[n]) for n in ('verification','reconstruction','replay','evaluation','exports')},
            'claims':{'training_verified_by':'project-operator','independent_third_party':False,
                      'signature_establishes':'publisher-endorsement-and-integrity-only','consumer_training_recomputed':False,
                      'factual_accuracy':'NOT_ESTABLISHED_BY_PROVENANCE_OR_TOKEN_LOSS'}}
    return validate(result)


def verify_payloads(statement,bundle,policy,directories,r):
    """Identity and actual payload integrity only; never claim local replay."""
    if type(policy) is not ReleasePublisherPolicy:raise EvidenceError('separate final release publisher policy required')
    policy.validate();receipt=verify_anchor(statement,bundle,policy)
    value=read_json(statement);validate(value)
    if digest(value)!=policy.statement_sha256 or value['registration_sha256']!=digest(r):raise EvidenceError('release differs from external selection')
    if value['code']['code_root']!=r['code_root'] or value['code']['revision']!=r['code_revision']:
        raise EvidenceError('released loader identity differs from registration')
    for phase in ('base','chat'):
        expected=value['models'][phase]
        if expected['repo']!=model_repo(r,phase,value['publication_id']):raise EvidenceError('model destination differs from registered attempt')
        verify_inventory(directories[phase],expected['files'])
        if read_json(directories[phase]/'registration.json')!=r:raise EvidenceError('downloaded registration differs')
        load_model(directories[phase]/'model',r,phase,expected_model_root=expected['model_root'])
    return {'schema':'ovl.release-payload-verification.v1','result':'PASS','release_sha256':digest(value),
            'scope':'publisher-identity-and-selected-model-payload-integrity','publisher':receipt,
            'attested_by':receipt['identity'],'performed_by':'verifier-operator',
            'locally_recomputed':['publisher_signature','model_file_hashes','model_tensor_roots','inference_configuration'],
            'locally_recomputed_training':False,'full_raw_reconstruction':'NOT_RUN','continuous_replay':'NOT_RUN',
            'public_download_performed':'NOT_RUN','factual_accuracy':'NOT_ESTABLISHED_BY_PROVENANCE_OR_TOKEN_LOSS'}


def prepare_payloads(r,report_directory,exports,source_checkout,output,*,source_statement,evidence_archive,publication='release-1'):
    """Build reviewable payloads after complete operator verification assertions.

    This builder rechecks weights/parents but does not replace the actual full
    verification command. The resulting unsigned inventory is not a publication.
    """
    values=reports(r,report_directory);archive_shape(evidence_archive);verify_inventory(report_directory,evidence_archive['inventory'])
    if digest(source_statement)!=r['source_statement_sha256']:raise EvidenceError('model card source statement differs from registration')
    evidence_url='https://huggingface.co/datasets/'+evidence_archive['repo']+'/tree/'+evidence_archive['revision']+'/'+evidence_archive['prefix']
    conversation_url='https://huggingface.co/datasets/'+source_statement['conversation']['repo']+'/tree/'+source_statement['conversation']['revision']
    retention_days=source_statement['archive']['retention_days_target']
    if output.exists():raise EvidenceError('release payload preparation requires fresh output')
    if read_json(exports/'export.json')!=values['exports']:raise EvidenceError('export report differs from selected verification evidence')
    license=confined(source_checkout,'LICENSE')
    if not license.is_file() or not license.read_text().startswith('GNU GENERAL PUBLIC LICENSE\nVersion 3,'):
        raise EvidenceError('preserved GPL-3.0 project license required')
    output.mkdir(parents=True,exist_ok=False)
    for phase in ('base','chat'):
        root=values['replay'][phase+'_model_root'];model=load_model(exports/phase,r,phase,expected_model_root=root)
        directory=output/phase;(directory/'model').mkdir(parents=True)
        for name in FILES:shutil.copyfile(confined(exports/phase,name),directory/'model'/name)
        write_json(directory/'registration.json',r)
        demonstrations=values['evaluation']['models'][phase]['demonstrations'];write_json(directory/'demonstrations.json',demonstrations)
        shutil.copyfile(license,directory/'LICENSE')
        repo=model_repo(r,phase,publication);parameters=sum(p.numel() for p in model.parameters())
        description='Wikipedia base model' if phase=='base' else 'conversational derivative of the Wikipedia base model'
        execution_note=('The reconstruction and replay were separate recorded executions. The final report assembly '
                        'rechecked their artifacts and evaluated the models; it did not rerun those computations.\n\n'
                        if values['verification']['schema']=='ovl.complete-computation-verification.v2' else '')
        card=f'''---
license: gpl-3.0
language: en
tags:
- openverifiablellm
- wikipedia
- provenance
{('- oasst1' + chr(10) + '- conversational' + chr(10)) if phase=='chat' else ''}---
# {repo.split('/')[-1]}

This {description} has {parameters:,} parameters. Training starts from regenerated
random initialization and covers exactly one complete pass of the registered
eligible English Wikipedia target stream. The conversational derivative continues
from the base state on the pinned public OpenAssistant/OASST1 selection.

The project operator reports complete raw-data reconstruction and continuous exact
replay of every update through both phases. The signed release inventory endorses
that operator report. It is not independent third-party verification. Run the
[complete verifier](https://github.com/AOSSIE-Org/OpenVerifiableLLM/blob/{r['code_revision']}/docs/COMPLETE_VERIFIER.md)
to recompute the declared computation on the stated compatible environment.
Checking the publisher signature alone does not replay training.

{execution_note}\
Registration SHA-256: `{digest(r)}`. Model tensor root: `{root}`.
The signed `release.json` binds payloads, source identity and public evidence.
Choose the release publisher trust policy independently; do not trust a policy
merely because it accompanies this model. Fixed-revision download verification
receipts are published separately in the project's public evidence.
[Verification reports and ancestry]({evidence_url}) are pinned at an immutable revision.

Use the tested custom loader from source revision `{r['code_revision']}`. Install
the locked CPU dependencies in `requirements/preparation.lock` and
`requirements/endorsement.lock`, then run offline with the downloaded files:

```bash
TOKENIZERS_PARALLELISM=false PYTHONPATH=src python -m ovl_pipeline.production_export infer \\
  --directory MODEL_DOWNLOAD/model --registration MODEL_DOWNLOAD/registration.json \\
  --registration-sha256 {digest(r)} --phase {phase} \\
  --prompt 'What is an encyclopedia?' --max-new-tokens 64
```

The command above checks internal model consistency only. Use the separately trusted
[public download verifier](https://github.com/AOSSIE-Org/OpenVerifiableLLM/blob/{r['code_revision']}/docs/RELEASE_VERIFICATION.md)
to authenticate the publisher and complete provenance. To reproduce the retained
demonstrations, use `--max-new-tokens 32` with each exact prompt:
`What is an encyclopedia?` and `Explain how rain forms.`

The initial release uses canonical FP32 safetensors, CPU greedy inference, the
explicit tokenizer and input template in `model/config.json`. Generated inputs,
output token IDs and runtime observations for public demonstrations are retained
in `demonstrations.json`. General Transformers or Ollama loading is untested.

The evaluation covers every eligible official held-out conversation target; its
exact token-loss arithmetic and results are in the public evidence. Token loss,
model provenance and reproducible output do not establish factual answer accuracy.
This small one-pass model has limited language and conversational capability.
No claim is made that a generated response is attributable to a particular article.

Project material and model artifacts use the included GPL-3.0 license. Source data
retain their respective licenses and attribution; see `MODEL_SOURCES.md` and the
complete public raw archive. Raw/prepared evidence has a {retention_days}-day best-effort public
retention target; final model and manifest retention is intended to be long-term.
These statements do not purchase indefinite storage or guarantee host availability.
'''
        (directory/'README.md').write_text(card)
        notices=f'''# Sources and attribution

Wikipedia source statement SHA-256: `{r['source_statement_sha256']}`.
Complete prepared-data manifest SHA-256: `{r['preparation_sha256']}`.
The public registration packet identifies the dated official dump, exact monolithic
file, complete checksums and retained acquisition metadata. Prepared article and
ledger records preserve article titles, page/revision IDs, timestamps and attribution,
history and revision URLs. Wikipedia text retains the applicable source licensing;
consult the retained notices and [Wikimedia terms](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use).

Conversation data: [OpenAssistant/OASST1]({conversation_url}).
The source statement pins the exact revision, complete train/validation Parquet files,
license and dataset card. The pinned source license and selection/loss-mask accounting
are retained in the public archive. The official validation split is held out.

The release evidence contains every trust link for both sources, transformations,
random initialization and phase transition. Source data are not relicensed merely
by inclusion in this model's provenance. The included GPL-3.0 license applies to
distributed project material and model artifacts; source notices remain applicable.
'''
        (directory/'MODEL_SOURCES.md').write_text(notices)
    return {phase:output/phase for phase in ('base','chat')}
