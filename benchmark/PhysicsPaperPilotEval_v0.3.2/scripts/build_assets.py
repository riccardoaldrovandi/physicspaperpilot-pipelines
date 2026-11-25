from pathlib import Path
import csv, json, re
ROOT=Path(__file__).resolve().parents[1]
KEY=ROOT/'02_private_answer_key'/'answer_key_private.csv'

def rd(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def wr(p,rows):
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else []);w.writeheader();w.writerows(rows)

# Curated lexical aliases. These are deterministic phrase variants, not semantic judgments.
A={
('PHY-170502355-Q02',1):['passes information from one calorimeter layer to the next','layer to layer information transfer','inter layer information transfer','passes information between successive calorimeter layers'],
('PHY-170502355-Q02',2):['inter layer correlations','correlations between successive calorimeter layers','sequential correlations across layers','sequential dependence across calorimeter layers'],
('PHY-170502355-Q03',1):['compared with geant4 showers','geant4 reference showers','comparison against geant4'],
('PHY-170502355-Q03',2):['shower shape distributions','shower shape variables','classification performance as a proxy','classifier performance proxy'],
('PHY-170502355-Q03',3):['some shower shape distributions are not well described','imperfect description of some distributions','limited precision across the full phase space','not accurate throughout the full phase space'],
('PHY-171210321-Q02',1):['photons','gamma'],('PHY-171210321-Q02',2):['positrons','positron','e plus'],('PHY-171210321-Q02',3):['positive pions','positively charged pions','pi plus','charged pions'],
('PHY-171210321-Q03',1):['100x to 1000x','100 to 1000 times','100x 1000x','cpu speed up 100 1000'],
('PHY-171210321-Q03',2):['up to 10 to 5','up to 100000x','10 to the 5','100000 times','gpu speed up 100000'],
('PHY-180500850-Q02',1):['stable particles from pythia6','stable pythia6 particles','pythia 6 stable particles'],
('PHY-180500850-Q02',2):['reconstructed particle candidates','particle flow candidates','reconstructed candidates'],
('PHY-180500850-Q02',3):['anti kt','anti kt algorithm','antikt','r 0 5','fastjet r 0 5'],
('PHY-180500850-Q03',1):['gan model for hadronic jet detector response','generative model for hadronic jets','detector response to hadronic jets'],
('PHY-180500850-Q03',2):['high fidelity','high fidelity modelling','faithful detector response modelling'],
('PHY-180500850-Q03',3):['several orders of magnitude faster','orders of magnitude speed up','speed increase of several orders'],
('PHY-180500850-Q03',4):['no single fixed speed up factor','not one universal speed up factor','speed up depends on configuration','does not report one fixed factor'],
('PHY-180701954-Q02',1):['geant4 simulated showers','showers simulated with geant4','geant 4 reference showers'],
('PHY-180701954-Q02',2):['incident electron beam','electron beam','electrons as incident particles'],
('PHY-180701954-Q02',3):['multi layer electromagnetic calorimeter','multilayer electromagnetic calorimeter','electromagnetic calorimeter with multiple layers'],
('PHY-180701954-Q03',1):['critic measures distance between real and generated distributions','critic estimates discrepancy between real and generated showers','distance between real and generated distributions'],
('PHY-180701954-Q03',2):['wasserstein distance guides adversarial training','wasserstein metric in generator critic training','wasserstein distance trains generator and critic'],
('PHY-210605285-Q02',1):['flow one models energy deposits per layer','flow i models layer wise energy depositions','layer wise energy deposition','energy deposition in each layer'],
('PHY-210605285-Q02',2):['flow two models shower shapes','flow ii models spatial shower shapes','shower shape distribution'],
('PHY-210605285-Q03',1):['explicit likelihood estimation','explicit probability density estimation','exact likelihood','density likelihood estimation'],
('PHY-210605285-Q03',2):['maximum likelihood training simplifies model selection','maximum likelihood objective supports model selection','likelihood based model selection'],
('PHY-210605285-Q03',3):['stable training','validation based epoch selection','selecting epochs using validation likelihood','more stable training and validation'],
('PHY-210605285-Q03',4):['mode collapse','fewer mode collapse problems','reduced artifacts compared with gans','avoids gan mode collapse'],
('PHY-220611898-Q02',1):['normalizing flows struggle with high dimensional data','normalizing flows do not scale well to high dimensions','scaling normalizing flows to higher dimensions is difficult'],
('PHY-220611898-Q02',2):['prohibitive training cost on the largest datasets','largest challenge datasets are too expensive to train on','training cost becomes prohibitive'],
('PHY-220611898-Q02',3):['score is the gradient of the log data density','gradient of log probability density','score function of the data distribution','learns the score of the data density'],
('PHY-220611898-Q03',1):['calochallenge 2022','fast calorimeter simulation challenge 2022','calorimeter simulation challenge datasets'],
('PHY-221006204-Q02',1):['central atlas electromagnetic calorimeter region','central region of the atlas ecal','central atlas electromagnetic calorimeter'],
('PHY-221006204-Q02',2):['photons','gamma'],('PHY-221006204-Q02',3):['multiple photon energies','different incident photon energies','more than one photon energy'],
('PHY-221006204-Q03',1):['large scale simulated event production','large samples of simulated events','large scale event simulation'],
('PHY-221006204-Q03',2):['simulation accuracy remains important','accuracy is still essential','maintaining accurate simulation'],
('PHY-221006204-Q03',3):['vae and gan approaches for atlas photon showers','variational autoencoders and gans for photon showers','photon shower modelling in atlas using vaes and gans'],
('PHY-221007430-Q02',1):['gumbolt dvae','gumbolt discrete variational autoencoder','gumbolt framework'],
('PHY-221007430-Q02',2):['conditioned on incident particle energy','conditioning on incoming particle energy','incident energy conditioning'],
('PHY-221007430-Q03',1):['geant4 version 10 2 0','geant4 10 2','geant 4 10 2 0'],
('PHY-221007430-Q03',2):['ftfp bert','ftfp bert physics list'],
('PHY-221007430-Q03',3):['electromagnetic physics package','electromagnetic physics list package','em physics package'],
('PHY-221115380-Q02',1):['high dimensional raw shower representation','showers are represented in a high dimensional space','high dimensional shower data'],
('PHY-221115380-Q02',2):['lower dimensional manifold','low dimensional manifold structure','manifold like low dimensional structure'],
('PHY-221115380-Q02',3):['dimensionality reduction before density estimation','learn the manifold before estimating density','reduces dimensionality prior to density estimation'],
('PHY-221115380-Q03',1):['histogram chi squared separation power','chi squared distance between histograms','histogram chi2 separation'],
('PHY-221115380-Q03',2):['high level features','high level shower features','reported on high level observables'],
('PHY-221115380-Q03',3):['smaller values are better','lower values indicate better agreement','lower is better'],
('PHY-230504847-Q02',1):['spatial point cloud','energy deposits as points','represents shower deposits as points','point cloud representation'],
('PHY-230504847-Q02',2):['not tied to a fixed voxel grid','does not rely on regular voxelization','independent of a fixed regular voxel layout'],
('PHY-230504847-Q02',3):['high granularity and geometry independence','geometry independent calorimeter simulation','highly granular geometry independent showers'],
('PHY-230504847-Q03',1):['total visible energy','sum of visible energy','e sum','esum'],
('PHY-230504847-Q03',2):['number of points per layer','points in each layer','n z i','nzi'],
('PHY-230504847-Q03',3):['post diffusion calibration','calibration after diffusion','used for the post diffusion calibration'],
('PHY-230511934-Q02',1):['high resolution creates memory constraints','memory cost becomes prohibitive at high resolution','memory limitations at higher resolution'],
('PHY-230511934-Q02',2):['teacher models are masked autoregressive flows','teacher maf models','mafs are the teachers'],
('PHY-230511934-Q02',3):['student models are inverse autoregressive flows','student iaf models','iafs are the students'],
('PHY-230511934-Q03',1):['unit normalized voxel energies in later layers','normalized voxel energies for subsequent layers','later layer voxel energies'],
('PHY-230511934-Q03',2):['conditioned on current and previous layer energy deposits','current and preceding layer deposited energies','energy deposits in current and previous layers'],
('PHY-230511934-Q03',3):['conditioned on incident energy','incident particle energy is a condition','incoming energy conditioning'],
('PHY-230511934-Q03',4):['previous layer normalized voxel pattern','normalized voxel pattern from the preceding layer','previous layer voxel energy pattern'],
('PHY-230511934-Q03',5):['one hot encoding of the layer','one hot layer encoding','layer index represented as one hot'],
('PHY-230803876-Q02',1):['cylindrical geometry optimizations','adaptations for cylindrical calorimeter geometry','optimization for cylindrical coordinates'],
('PHY-230803876-Q02',2):['geometry latent mapping','glam'],
('PHY-230803876-Q03',1):['delta 10 to minus 6','delta equals 1e 6','small delta offset avoids boundary discontinuities','avoids discontinuities at zero and one'],
('PHY-230803876-Q03',2):['mean centered','subtracting the mean','centered by the mean'],
('PHY-230803876-Q03',3):['divided by the standard deviation','standard deviation normalization','zero mean and unit variance','standardized to unit variance'],
('PHY-230811700-Q02',1):['three dimensional coarse graining','3d coarse graining','coarse graining in three dimensions'],
('PHY-230811700-Q02',2):['3d coarse graining gave poorer quality','lower quality than the two dimensional options','worse results than the 2d choices'],
('PHY-230811700-Q02',3):['downsides of both two dimensional choices','drawbacks of both 2d options','disadvantages of choices a and b'],
('PHY-230811700-Q03',1):['high dimensionality makes training and generation slow','poor scaling of training and generation time','computational cost grows with shower dimensionality'],
('PHY-230811700-Q03',2):['coarse to fine upsampling','upsampling from coarse showers to fine showers','coarse grained showers are upsampled'],
('PHY-230811700-Q03',3):['reduced computation memory and generation time','lower compute and memory requirements','reduced computational and memory cost'],
('PHY-230811700-Q03',4):['variation among generated upsampled showers','diversity of upsampled samples','variation across upsampled shower samples'],
('PHY-240211575-Q02',1):['1478 edges','one thousand four hundred seventy eight edges','photon graph has 1478 edges'],
('PHY-240211575-Q02',2):['2210 edges','two thousand two hundred ten edges','pion graph has 2210 edges'],
('PHY-240211575-Q03',1):['eta and phi are standardized to zero mean and unit standard deviation','eta phi mean zero standard deviation one','coordinates eta and phi normalized to mean zero and std one'],
('PHY-240211575-Q03',2):['energy normalized to incident energy','voxel energy divided by incident energy','relative to the incoming particle energy'],
('PHY-240211575-Q03',3):['factor 3 1 for photons','photon normalization factor 3 1','photons use a factor of 3 1'],
('PHY-240211575-Q03',4):['factor 6 4 for pions','pion normalization factor 6 4','pions use a factor of 6 4'],
('PHY-241021611-Q02',1):['four datasets','4 datasets'],('PHY-241021611-Q02',2):['a few hundred voxels','hundreds of voxels','several hundred voxels'],('PHY-241021611-Q02',3):['a few tens of thousands of voxels','tens of thousands of voxels','several ten thousand voxels'],
('PHY-241021611-Q03',1):['shared basis for comparison','common basis for comparing models','same datasets enable fair comparison'],
('PHY-241021611-Q03',2):['increasing dimensionality','datasets span different dimensions','from lower to higher dimensionality'],
('PHY-241021611-Q03',3):['multiple state of the art generative models','several state of the art generative models','compares many generative models'],
('PHY-241021611-Q03',4):['challenge report rather than a new model','reports a community challenge','does not propose a single new generative model'],
}


# Additional normalization-aware variants for scientific symbols and numbers.
A.update({
 ('PHY-171210321-Q03',2): A.get(('PHY-171210321-Q03',2),[]) + ['10 to 5','10 power 5','one hundred thousand','100000'],
 ('PHY-180500850-Q02',3): A.get(('PHY-180500850-Q02',3),[]) + ['anti kt with radius 0 5','anti kt r equals 0 5'],
 ('PHY-230803876-Q03',1): A.get(('PHY-230803876-Q03',1),[]) + ['delta 10 to 6','delta equals 10 to 6','delta is 0 000001','delta 1e 6'],
 ('PHY-240211575-Q02',1): A.get(('PHY-240211575-Q02',1),[]) + ['photon graph 1478'],
 ('PHY-240211575-Q02',2): A.get(('PHY-240211575-Q02',2),[]) + ['pion graph 2210'],
})


A.update({
 ('PHY-170502355-Q02',1): A[('PHY-170502355-Q02',1)] + ['carries information from one calorimeter layer to the next'],
 ('PHY-170502355-Q02',2): A[('PHY-170502355-Q02',2)] + ['physical sequential dependence between layers','sequential dependence between layers'],
 ('PHY-170502355-Q03',1): A[('PHY-170502355-Q03',1)] + ['compares generated showers with geant4'],
 ('PHY-170502355-Q03',3): A[('PHY-170502355-Q03',3)] + ['some distributions are not described well','precision across the full phase space remains challenging'],
 ('PHY-171210321-Q03',2): A[('PHY-171210321-Q03',2)] + ['10 to the 5','approximately 10 5'],
 ('PHY-180500850-Q03',1): A[('PHY-180500850-Q03',1)] + ['gan based deep neural networks model hadronic jet detector response'],
 ('PHY-180500850-Q03',4): A[('PHY-180500850-Q03',4)] + ['abstract does not give a single universal numerical factor'],
 ('PHY-180701954-Q02',1): A[('PHY-180701954-Q02',1)] + ['geant4 simulations'],
 ('PHY-180701954-Q03',1): A[('PHY-180701954-Q03',1)] + ['distribution distance objective','critic and generator use a distribution distance','distance between real and generated distributions'],
 ('PHY-180701954-Q03',2): A[('PHY-180701954-Q03',2)] + ['wasserstein distance objective','wasserstein distance used to guide critic and generator'],
 ('PHY-210605285-Q03',1): A[('PHY-210605285-Q03',1)] + ['explicit density estimator','flows train an explicit density estimator'],
 ('PHY-210605285-Q03',2): A[('PHY-210605285-Q03',2)] + ['maximum likelihood objective','straightforward maximum likelihood objective'],
 ('PHY-210605285-Q03',3): A[('PHY-210605285-Q03',3)] + ['training and model selection more stable','stable and transparent model selection'],
 ('PHY-220611898-Q02',1): A[('PHY-220611898-Q02',1)] + ['difficult to scale to higher dimensional datasets'],
 ('PHY-220611898-Q02',2): A[('PHY-220611898-Q02',2)] + ['training them on the larger calochallenge datasets would be prohibitively expensive'],
 ('PHY-221006204-Q02',3): A[('PHY-221006204-Q02',3)] + ['photons of various energies','various photon energies'],
 ('PHY-221006204-Q03',2): A[('PHY-221006204-Q03',2)] + ['accurate simulated event samples','large scale production of accurate simulated events'],
 ('PHY-221006204-Q03',3): A[('PHY-221006204-Q03',3)] + ['deep generative models for photon responses in the electromagnetic calorimeter'],
 ('PHY-230504847-Q02',2): A[('PHY-230504847-Q02',2)] + ['fixed dense voxel grid','rather than requiring a fixed dense voxel grid'],
 ('PHY-230504847-Q02',3): A[('PHY-230504847-Q02',3)] + ['geometry independent treatment','highly granular showers'],
 ('PHY-230504847-Q03',3): A[('PHY-230504847-Q03',3)] + ['used for post diffusion calibration'],
 ('PHY-230511934-Q02',1): A[('PHY-230511934-Q02',1)] + ['higher detector resolutions leads to prohibitive memory constraints'],
 ('PHY-230511934-Q02',2): A[('PHY-230511934-Q02',2)] + ['teacher masked autoregressive flows'],
 ('PHY-230511934-Q02',3): A[('PHY-230511934-Q02',3)] + ['student inverse autoregressive flows'],
 ('PHY-230511934-Q03',1): A[('PHY-230511934-Q03',1)] + ['unit normalized voxel energy distribution in each layer after the first'],
 ('PHY-230511934-Q03',2): A[('PHY-230511934-Q03',2)] + ['deposited energies in the current and preceding layers'],
 ('PHY-230511934-Q03',4): A[('PHY-230511934-Q03',4)] + ['normalized voxel energy pattern in the preceding layer'],
 ('PHY-230803876-Q03',1): A[('PHY-230803876-Q03',1)] + ['delta 10 to minus 6','delta 10 6','delta equals 0 000001'],
 ('PHY-230803876-Q03',3): A[('PHY-230803876-Q03',3)] + ['zero mean and unit standard deviation','centered and scaled to have zero mean and unit standard deviation'],
 ('PHY-230811700-Q02',1): A[('PHY-230811700-Q02',1)] + ['three dimensional coarse grainings'],
 ('PHY-230811700-Q02',2): A[('PHY-230811700-Q02',2)] + ['produced poorer quality results','poorer quality results'],
 ('PHY-230811700-Q03',1): A[('PHY-230811700-Q03',1)] + ['training and generation costs that scale poorly'],
 ('PHY-230811700-Q03',3): A[('PHY-230811700-Q03',3)] + ['reduce computational cost memory requirements and generation time'],
 ('PHY-230811700-Q03',4): A[('PHY-230811700-Q03',4)] + ['retaining shower variation'],
 ('PHY-241021611-Q02',1): A[('PHY-241021611-Q02',1)] + ['four calorimeter shower datasets'],
 ('PHY-241021611-Q03',1): A[('PHY-241021611-Q03',1)] + ['common datasets spanning different dimensionalities','same datasets for systematic comparison'],
 ('PHY-241021611-Q03',2): A[('PHY-241021611-Q03',2)] + ['different dimensionalities','datasets span increasing dimensionalities'],
 ('PHY-241021611-Q03',3): A[('PHY-241021611-Q03',3)] + ['different generative approaches','multiple generative approaches'],
 ('PHY-241021611-Q03',4): A[('PHY-241021611-Q03',4)] + ['not a single method paper','not a single method model'],
})


A.update({
 ('PHY-171210321-Q03',2): A[('PHY-171210321-Q03',2)] + ['up to approximately 105 x','approximately 105 x on gpu'],
 ('PHY-180500850-Q03',1): A[('PHY-180500850-Q03',1)] + ['gan based deep neural networks can model hadronic jet detector response'],
 ('PHY-180701954-Q03',2): A[('PHY-180701954-Q03',2)] + ['wasserstein distance'],
 ('PHY-221006204-Q03',3): A[('PHY-221006204-Q03',3)] + ['the paper explores vaes and gans','vaes and gans for photon responses'],
 ('PHY-221115380-Q03',1): A[('PHY-221115380-Q03',1)] + ['histogram 2 separation power','chi 2 separation power'],
 ('PHY-230511934-Q03',3): A[('PHY-230511934-Q03',3)] + ['incident energy'],
})

# v0.3.1: aliases for the key point added to PHY-240211575-Q03 (logit step of the
# voxel-energy preprocessing, Sec. II.B of the paper). See CHANGELOG_v0.3.1.md.
A[('PHY-240211575-Q03',5)] = ['logit transform','logit transformation','logit transformed','logit function',
                              'logit transform with alpha 10 to minus 6','logit followed by standardization']

rows=rd(KEY)
new=[]
lex=[]
evidence_template=[]
for r in rows:
    points=json.loads(r['key_points'])
    ev=json.loads(r['evidence'])
    # Each original evidence record is one gold evidence group. When its section
    # field lists several locations separated by '/', they are treated as
    # acceptable alternative source headings for that evidence group.
    evidence_groups=[]
    flat_sections=[]
    for j,item in enumerate(ev,1):
        sec=item.get('section','').strip()
        alternatives=[x.strip() for x in re.split(r'\s*/\s*',sec) if x.strip()]
        if not alternatives and sec: alternatives=[sec]
        evidence_groups.append({
            'evidence_id':f"{r['question_id']}-E{j:02d}",
            'accepted_sections':alternatives,
            'detail':item.get('detail','')
        })
        for part in alternatives:
            if part not in flat_sections: flat_sections.append(part)
    if r['question_id']=='PHY-180701954-Q03':
        r['gold_answer']='It uses the Wasserstein distance as a distribution-distance objective to guide the critic and generator, rather than relying only on a binary real/fake classification objective.'
        r['reference_answers_json']=json.dumps([r['gold_answer']],ensure_ascii=False)
    if r['question_id']=='PHY-221006204-Q03':
        r['gold_answer']='The ATLAS physics programme needs large-scale production of accurate simulated event samples. The paper explores VAEs and GANs as fast modelling approaches for photon showers in ATLAS, aiming to model photon responses in the electromagnetic calorimeter more efficiently than relying exclusively on computationally expensive detailed simulation.'
        r['reference_answers_json']=json.dumps([r['gold_answer']],ensure_ascii=False)
    if r['question_id']=='PHY-230504847-Q03':
        r['gold_answer']='It generates the total visible calorimeter energy E_sum and the number of points per layer N_z,i; these quantities are used for post-diffusion calibration.'
        r['reference_answers_json']=json.dumps([r['gold_answer']],ensure_ascii=False)
    r['gold_evidence_groups_json']=json.dumps(evidence_groups,ensure_ascii=False)
    r['gold_evidence_sections_json']=json.dumps(flat_sections,ensure_ascii=False)
    r['evidence_scoring_level']='section_heading_only'
    if r['question_type']=='multiple_choice':
        r['scoring_method_no_llm']='accuracy+evidence_section_f1'
        r['evaluation_method']='exact_accuracy_and_section_citation'
    elif r['question_type']=='short_answer':
        r['scoring_method_no_llm']='normalized_em+token_f1+keypoint_coverage+evidence_section_f1'
        r['evaluation_method']='automatic_reference_similarity_and_keypoint_coverage'
    else:
        r['scoring_method_no_llm']='token_f1+rougeL+keypoint_coverage+evidence_section_f1'
        r['evaluation_method']='automatic_reference_similarity_and_keypoint_coverage'
    r['passage_evidence_status']='not_annotated'
    new.append(r)
    if r['question_type'] in ('short_answer','open_ended'):
        for idx,kp in enumerate(points,1):
            aliases=[kp]
            aliases.extend(A.get((r['question_id'],idx),[]))
            # Basic spelling/format variants.
            aliases.extend([kp.replace('modelling','modeling').replace('normalised','normalized'),
                            kp.replace('calorimeter layers','layers of the calorimeter')])
            ded=[]; seen=set()
            for a in aliases:
                a=a.strip()
                na=' '.join(re.sub(r'[^a-z0-9]+',' ',a.lower()).split())
                if na and na not in seen: ded.append(a);seen.add(na)
            lex.append({
                'question_id':r['question_id'],
                'keypoint_id':f"{r['question_id']}-KP{idx:02d}",
                'key_point':kp,
                'accepted_phrases_json':json.dumps(ded,ensure_ascii=False),
                'matching_rule':'any_normalized_phrase_or_conservative_anchor_token_match',
                'validation_status':'candidate_lexicon_requires_domain_review'
            })
    for j,item in enumerate(ev,1):
        evidence_template.append({
            'evidence_id':f"{r['question_id']}-E{j:02d}",
            'question_id':r['question_id'],'arxiv_id':r['arxiv_id'],'paper_version':r['paper_version'],
            'section_label_from_current_gold':item.get('section',''),
            'current_evidence_description':item.get('detail',''),
            'page_number_to_annotate':'','paragraph_or_span_id_to_annotate':'','exact_quote_to_annotate':'',
            'figure_table_equation_id_to_annotate':'','annotator':'','independent_check_by':'',
            'validation_status':'needs_precise_page_paragraph_or_span_annotation'
        })
wr(KEY,new)
wr(ROOT/'02_private_answer_key'/'keypoints_lexicon.csv',lex)
wr(ROOT/'02_private_answer_key'/'evidence_annotations_template.csv',evidence_template)
print('Updated answer key',len(new),'keypoint rows',len(lex),'evidence annotations',len(evidence_template))
