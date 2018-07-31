def extract(data, length_of_data):
    """
    @param number_of_source: number of sources
    @param length_of_data: lenght of data
    
    Returns;
    index: [[source]]
    claim: [[fact]]
    index[i, k] is the source of fact claim[i, k]
    """
    sources_of_objects = []
    facts_of_objects = []
    for i in range(length_of_data):
        sources_of_objects.append(list(data[i][:, 0]))
        facts_of_objects.append(data[i][:, 1])
    return [sources_of_objects, facts_of_objects]


def update_source(sources_of_objects, facts_of_objects, s_set, number_of_sources, length_of_data):
    """
    @docstring
    --------------------------------------------------------------------------
    count -> In our case, the number of population each contributor provides.
             According to paper, the number of fact each source provides.
    --------------------------------------------------------------------------
    s_set -> List of confidence of each fact
    --------------------------------------------------------------------------
    
    """
    trustworthiness_vector = np.zeros(number_of_sources)
    tau_vec = np.zeros(number_of_sources)
    count = np.zeros(number_of_sources)

    # Equation 1 in Paper starts.
    for i in range(length_of_data):
        trustworthiness_vector[facts_of_objects[i]] = trustworthiness_vector[facts_of_objects[i]] + s_set[i]
        count[facts_of_objects[i]] = count[facts_of_objects[i]] + 1

    trustworthiness_vector[count > 0] = trustworthiness_vector[count > 0]/count[count > 0]
    # Equation 1 in Paper end.

    # trustworthiness_vector >= 1 means, return index of vector elements where vector element bigger than 1

    tau_vec[trustworthiness_vector >= 1] = np.log(1e10) 
    # print(trustworthiness_vector[trustworthiness_vector >= 1])

    # Equation 4 in Paper.
    tau_vec[trustworthiness_vector < 1] = -np.log(1-trustworthiness_vector[trustworthiness_vector < 1])
    return tau_vec


def update_claim(facts_of_objects, sources_of_objects, tau_vector, length_of_data, rho_constant, gamma_constant):
    """
    --------------------------------------------------------------------------
    s_set -> List of confidence of each fact
    --------------------------------------------------------------------------
    s_vec -> Stores each fact's confidence value related to that object.
    --------------------------------------------------------------------------
    """
    s_set = []
    for i in range(length_of_data):
        claim_set = list(set(facts_of_objects[i]))
        # Equation 5, Lemma 1 -> sigma_i = ....
        sigma_i = np.zeros(len(claim_set))
        # Each index of s_set is s_vec.
        s_vec = np.zeros(len(facts_of_objects[i]))

        # Lemma 1 calculation starts here.
        for j in range(len(claim_set)):
            # this line does sigma(f) = where w is in W(f): sum(tau(f)) for an object
            sigma_i[j] = sum(tau_vector[sources_of_objects[i]][facts_of_objects[i]==claim_set[j]])
            # print(tau_vector[sources_of_objects[i]][facts_of_objects[i]==claim_set[j]])
            print(str(len(tau_vector[sources_of_objects[i]])) + " - " + str(len(facts_of_objects[i]==claim_set[j])))

        # Lemma 1 calculation end here.

        tmp_i = np.copy(sigma_i)
        # Equation 6 calculation starts here.
        for j in range(len(claim_set)):
            tmp_i[j] = (1-rho_constant) * sigma_i[j] + rho_constant * sum((np.exp(-abs(claim_set-claim_set[j])))*sigma_i)
            #tmp_i[j] = (1+rho)*sigma_i[j] + rho*sum(-sigma_i)
            s_vec[facts_of_objects[i]==claim_set[j]] = 1/(1 + np.exp(-gamma_constant*tmp_i[j]))
        # Equation 6 calculation end here.

        s_set.append(s_vec)

    return(s_set)


def TruthFinder(data, number_of_sources, length_of_data, tolerans=0.1, max_iteration=10):
    '''
    @docstring
    -----------------------------------------------------------------
    facts_of_objects -> i.e. Newyork:
    facts_of_objects[0] = [population_of_Newyork_from_contributorX, population_of_Newyork_from_contibutorY, ...]:
    Where population_of_Country_from_contributorZ is type of integer.
    -----------------------------------------------------------------
    sources_of_objects -> Source:
    sources_of_objects i.e. contributors for Newyork:
    sources_of_objects[0] = [contributorX_id, contributorY_id, contributorZ_id, ....]:
    Where contributorX_id is type of integer.
    -----------------------------------------------------------------
    data is a list of pairs where first item is source of object and second item is fact of object.
    data[0] = [source_of_object, fact_of_object] -> is a single object
    -----------------------------------------------------------------
    length_of_data -> number of objects in data list
    -----------------------------------------------------------------
    tau_vector -> Trustworthiness score of w, 
    Where w is source, in our case, w is a single contributor.
    -----------------------------------------------------------------
    truth_vector-> Fact list that provides maximum truth score for each object.
    -----------------------------------------------------------------
    '''
    # If change in truth vector is smaller than error, finish iteration.
    error = 99
    # Retrieving sources and facts
    sources_of_objects, facts_of_objects = extract(data, length_of_data)
    # Initializing iteration.
    iteration = 0
    # Equation 4 in Paper.
    tau_vector = -np.log(1-np.ones(number_of_sources)*0.9)
    # Initializing truth vector
    truth_vector = np.zeros(length_of_data)
    # Gamma and rho values are represented in Paper, Section 4.2.3 Parameter Sensitivity.
    # Used in Equation 6 in Paper.
    rho_constant = 0.5
    # Used in Equation 8 in Paper.
    gamma_constant = 0.3

    while (error > tolerans) & (iteration < max_iteration):
        # iteration++
        iteration = iteration + 1
        # tau_old will be used in cosine similarity
        tau_old = np.copy(tau_vector)
        # list of confidence of facts -> s_set
        s_set = update_claim(facts_of_objects, sources_of_objects, tau_vector, length_of_data, rho_constant, gamma_constant)
        # tau_vector -> check docstring
        tau_vector = update_source(facts_of_objects, sources_of_objects, s_set, number_of_sources, length_of_data)
        # calculating cosine similarity
        error = 1 - np.dot(tau_vector, tau_old)/(la.norm(tau_vector)*la.norm(tau_old))
        print(iteration, error)

    for i in range(length_of_data):
        #  Takes maximum confidence score for each objects,
        truth_vector[i] = facts_of_objects[i][np.argmax(s_set[i])]

    return([truth_vector, tau_vector])
