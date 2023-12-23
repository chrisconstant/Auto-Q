from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

rouge_scores = scorer.score('How can we ensure that the maintenance strategies are tailored to the specific needs of the organization and the equipment being maintained, and what are the potential challenges associated with doing so?',
                      'How can we ensure that the maintenance schedules are optimized to minimize downtime and maximize productivity, and what are the potential challenges associated with doing so?')

#print (rouge_scores)

rouge_scores = scorer.score('Can you provide more details about the potential difficulties in integrating the anomaly model with existing systems, and how can we ensure a smooth integration process?',
                      'Can you provide more details about the specific steps involved in reviewing the current maintenance schedule and procedures, and how can we ensure that the anomaly model is integrated effectively')

#print (rouge_scores)

qarray = [
    'Can you provide more details about the specific bearing, gear, shaft, and coupling failure modes that are most critical in wind turbine gearboxes, and how do they typically manifest in terms of vibration, temperature, pressure, speed, and sound?',
    'How do the specific components of the wind turbine gearbox, such as the bearings, gears, shafts, and couplings, affect the likelihood and severity of these critical failure modes, and how can we optimize their design and maintenance to reduce the risk of failure?',
    'How can we ensure that the anomaly model is trained on a representative dataset that includes examples of all the critical failure modes, and what are the potential consequences of biased or incomplete training data?',
    'Can you provide more details about the potential regulatory or compliance requirements that we need to adhere to when implementing the anomaly model, such as data privacy and security regulations?',
    'How can we use feedback mechanisms to continuously improve the performance of the anomaly model, such as incorporating user feedback or updating the model with new data, and how does the sampling rate affect the performance of the feedback mechanisms?',
    'Can you provide more details about the specific maintenance strategies that can be employed to prevent wind turbine gearbox failures, such as lubrication, cooling, and control system maintenance, and how can we use the anomaly model to optimize these strategies?',
    'Can you provide more details about the specific failure modes that can occur in the bearings, gears, shafts, couplings, lubrication system, and cooling system of a wind turbine gearbox, and how do they typically manifest in terms of vibration, temperature, pressure, speed, and sound?',
    'How do the specific components of the wind turbine gearbox, such as the bearings, gears, shafts, couplings, lubrication system, and cooling system, affect the likelihood and severity of failure modes, and how can we optimize their design and maintenance to reduce the risk of failure?',
    'Can you provide more details about the specific bearing, gear, shaft, and coupling failure modes that are most critical in wind turbine gearboxes, and how do they typically manifest in terms of vibration, temperature, pressure, speed, and sound?'
]

for i in range(len(qarray)):
    for j in range(len(qarray)):
        if i < j:
            score = scorer.score(qarray[i],qarray[j])
            if score['rougeL'].fmeasure > 0.70:
                print ('source ..... ---->', score['rougeL'].fmeasure)
                print (qarray[i], '\n', qarray[j])