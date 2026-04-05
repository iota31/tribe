"""Internal controlled paired dataset for manipulation detection benchmarking.

Contains 25 pairs of texts on the same topic, one manipulative and one neutral.
Each pair covers a distinct policy/social topic. The manipulative versions use
fear appeals, loaded language, urgency, us-vs-them framing, and emotional
manipulation. The neutral versions present the same facts objectively with
nuance, sourcing, and balanced perspective.

No download is needed --- the dataset is embedded in this module.
"""

from __future__ import annotations

_PAIRS: list[dict] = [
    # --- Pair 1: Vaccine safety ---
    {
        "pair_id": 1,
        "topic": "Vaccine safety",
        "manipulative": (
            "They want you to inject your children with chemicals they refuse to fully "
            "disclose. The pharmaceutical giants raking in billions have bought the silence "
            "of regulators who are supposed to protect your family. Every year, thousands of "
            "parents report devastating reactions --- seizures, autoimmune collapse, children "
            "who were healthy one day and struggling the next. Yet anyone who dares question "
            "the official narrative is labeled anti-science and silenced. Ask yourself: if "
            "these products are truly safe, why did manufacturers demand legal immunity from "
            "lawsuits? Why are injury compensation courts buried in backlogs? Your child is "
            "not a line item on a corporate balance sheet. The window to protect them is "
            "closing. Once the damage is done, no amount of money will bring back what was "
            "lost. Wake up before it is too late."
        ),
        "neutral": (
            "Vaccine safety is monitored through multiple independent systems. In the United "
            "States, the Vaccine Adverse Event Reporting System (VAERS) collects voluntary "
            "reports, while the Vaccine Safety Datalink (VSD) conducts active surveillance "
            "across several large health systems. Serious adverse events, while documented, "
            "remain rare relative to the hundreds of millions of doses administered annually. "
            "The National Vaccine Injury Compensation Program has paid out approximately $4.7 "
            "billion since 1988, covering cases where a causal link was established or "
            "presumed. Researchers continue to study both short-term and long-term outcomes. "
            "A 2021 meta-analysis in The Lancet reviewing 12 million vaccine recipients found "
            "that severe adverse events occurred at rates consistent with background population "
            "rates for most conditions. Ongoing transparency about risks and benefits helps "
            "maintain public trust in immunization programs."
        ),
    },
    # --- Pair 2: Climate change ---
    {
        "pair_id": 2,
        "topic": "Climate change",
        "manipulative": (
            "Our planet is dying, and the elites who caused this crisis are doing nothing "
            "while ordinary people suffer. Record-breaking fires, floods that swallow entire "
            "towns, heat waves killing the elderly in their own homes --- this is not some "
            "distant future scenario, this is happening right now to people like you and your "
            "neighbors. The fossil fuel industry has known for decades and spent billions to "
            "cover it up, buying politicians and planting fake scientists to confuse you. "
            "They are gambling with your children's future for quarterly profits. If we do "
            "not act with radical urgency in the next few years, feedback loops will make "
            "recovery impossible. There will be no second chances. Every day of inaction is "
            "a betrayal of every generation that comes after us. You are either part of the "
            "solution or you are complicit in the destruction."
        ),
        "neutral": (
            "Global average temperatures have risen approximately 1.1 degrees Celsius above "
            "pre-industrial levels, according to the IPCC Sixth Assessment Report. Climate "
            "models project continued warming under most emission scenarios, with associated "
            "increases in extreme weather frequency and intensity. The economic costs of "
            "climate-related disasters reached $313 billion globally in 2022, per Munich Re "
            "data. Mitigation strategies range from renewable energy deployment and carbon "
            "pricing to land-use reform and energy efficiency standards. Adaptation measures "
            "include infrastructure hardening, early warning systems, and agricultural "
            "diversification. Progress is uneven: while renewable capacity additions reached "
            "record levels in 2023, global CO2 emissions also hit a new high. The transition "
            "involves complex trade-offs between economic development, energy access, and "
            "emission reduction, with disproportionate impacts on developing nations."
        ),
    },
    # --- Pair 3: Immigration ---
    {
        "pair_id": 3,
        "topic": "Immigration",
        "manipulative": (
            "While you struggle to make ends meet, your government is rolling out the red "
            "carpet for people who broke the law to get here. They flood our hospitals, "
            "overcrowd our schools, and drive down wages for hardworking citizens. Entire "
            "neighborhoods have been transformed beyond recognition, and if you dare say "
            "anything about it, you are smeared as a bigot. The political class wants cheap "
            "labor and guaranteed votes, and they do not care that your community is paying "
            "the price. Violent crime committed by illegal immigrants fills the local news, "
            "but the national media buries these stories to protect their open-borders agenda. "
            "This is not about compassion --- it is about power. They are replacing your voice "
            "with people who will be grateful enough to vote the right way. How much more are "
            "you willing to lose before you fight back?"
        ),
        "neutral": (
            "Immigration policy involves balancing economic, humanitarian, and security "
            "considerations. The Congressional Budget Office estimated that the 2013 "
            "comprehensive immigration reform bill would have increased real GDP by 3.3 "
            "percent over two decades. Research from the National Academy of Sciences found "
            "that first-generation immigrants cost state and local governments more than they "
            "contribute in taxes, but their children and grandchildren become among the "
            "strongest net fiscal contributors. Labor economists debate wage effects: some "
            "studies show small negative impacts on native workers without high school "
            "diplomas, while others find negligible effects. Unauthorized border crossings "
            "fluctuate significantly with economic conditions and policy changes in both "
            "origin and destination countries. Most policy experts across the political "
            "spectrum agree that the current system needs reform, though they differ "
            "substantially on the specifics of enforcement, pathways, and quotas."
        ),
    },
    # --- Pair 4: Government surveillance ---
    {
        "pair_id": 4,
        "topic": "Government surveillance",
        "manipulative": (
            "Every text you send, every call you make, every website you visit --- they are "
            "watching all of it. The surveillance state has grown into a monster that the "
            "founders of this country would never have tolerated. Intelligence agencies "
            "operate with virtually no oversight, hoovering up the private data of millions "
            "of innocent citizens under the thin pretense of national security. Whistleblowers "
            "who tried to expose this were hunted down and destroyed. Your phone is a tracking "
            "device in your pocket that you paid for yourself. They know where you go, who "
            "you talk to, what you believe. And now they want to expand these powers even "
            "further. If you think you have nothing to hide, remember that every authoritarian "
            "regime in history started by convincing people that surveillance was for their "
            "own protection. By the time you realize you should have been afraid, it will be "
            "too late to resist."
        ),
        "neutral": (
            "Government surveillance programs operate under a framework of statutes, executive "
            "orders, and judicial oversight. The Foreign Intelligence Surveillance Court "
            "(FISC) reviews warrant applications for electronic surveillance in national "
            "security cases, approving approximately 99 percent of requests historically. "
            "Following the Snowden disclosures in 2013, the USA FREEDOM Act of 2015 ended "
            "bulk collection of domestic phone metadata by the NSA, requiring more targeted "
            "requests. A 2020 report by the Privacy and Civil Liberties Oversight Board found "
            "both legitimate intelligence value and civil liberties concerns in remaining "
            "programs. Technology companies have expanded encryption and transparency reports "
            "in response. Legal scholars continue to debate the Fourth Amendment implications "
            "of modern surveillance capabilities. The balance between security and privacy "
            "remains an active area of policy development, with perspectives varying widely "
            "among security professionals, civil liberties advocates, and the judiciary."
        ),
    },
    # --- Pair 5: Food safety ---
    {
        "pair_id": 5,
        "topic": "Food safety",
        "manipulative": (
            "The food on your table is poisoning you and your family, and the agencies "
            "supposed to protect you are in the pocket of the corporations doing it. "
            "Pesticide residues linked to cancer coat the produce in your grocery store. "
            "Processed foods are engineered to be addictive, loaded with chemicals banned "
            "in other countries but perfectly legal here because industry lobbyists wrote "
            "the rules. The FDA rubber-stamps ingredients that European regulators rejected "
            "years ago. Your children are eating food dyes linked to behavioral disorders "
            "every single day. Meanwhile, executives at these food conglomerates feed their "
            "own families organic meals from private gardens. They know exactly what they are "
            "selling you, and they do not care. Every meal you serve without knowing what is "
            "really in it is a gamble with your family's health. Stop trusting the labels "
            "they designed to mislead you."
        ),
        "neutral": (
            "Food safety regulation in the United States involves multiple agencies, "
            "primarily the FDA and USDA. The FDA's pesticide residue monitoring program "
            "found that 99 percent of domestic samples in 2022 were within established "
            "tolerances, though critics note that tolerance levels themselves are debated. "
            "Regulatory approaches differ internationally: the EU bans several food additives "
            "permitted in the US under the generally recognized as safe (GRAS) framework. "
            "A 2023 FDA review identified approximately 10,000 chemicals allowed in the US "
            "food supply, many approved decades ago under different evidentiary standards. "
            "Researchers at Tufts University found that ultra-processed foods now comprise "
            "about 58 percent of American caloric intake, with epidemiological studies "
            "associating high consumption with increased health risks. Food labeling "
            "requirements have expanded over time, and ongoing debates focus on whether "
            "current standards adequately inform consumers about ingredients and sourcing."
        ),
    },
    # --- Pair 6: Education funding ---
    {
        "pair_id": 6,
        "topic": "Education funding",
        "manipulative": (
            "They are deliberately starving our schools to keep your children ignorant and "
            "compliant. While politicians send their own kids to elite private academies, "
            "public school classrooms crumble, teachers buy supplies out of their own pockets, "
            "and students share outdated textbooks. This is not an accident --- it is a "
            "strategy. An uneducated population is easier to control, easier to manipulate, "
            "easier to exploit. Every dollar they divert from education into tax breaks for "
            "the wealthy is a dollar stolen from your child's future. The achievement gap is "
            "not a natural outcome; it is manufactured inequality designed to ensure that the "
            "children of working families never rise above their station. If we do not demand "
            "massive investment in our schools right now, we condemn an entire generation to "
            "a life of diminished opportunity. Your children deserve better than the crumbs "
            "they are being offered."
        ),
        "neutral": (
            "Education funding in the United States totaled approximately $870 billion in "
            "2022, with about 47 percent from state sources, 45 percent from local property "
            "taxes, and 8 percent from federal programs. Per-pupil spending varies "
            "significantly: the highest-spending states allocate roughly three times more than "
            "the lowest per student. Research on the relationship between spending and outcomes "
            "is mixed. A landmark study by Jackson, Johnson, and Persico (2016) found that a "
            "10 percent increase in per-pupil spending led to measurably higher adult earnings "
            "and educational attainment for students from low-income families. However, other "
            "researchers emphasize that how money is spent matters as much as the total amount. "
            "School finance reform lawsuits have been filed in nearly every state, with courts "
            "ruling in favor of plaintiffs in about two-thirds of cases. Policy discussions "
            "continue to weigh equity, adequacy, and efficiency in allocation models."
        ),
    },
    # --- Pair 7: Healthcare costs ---
    {
        "pair_id": 7,
        "topic": "Healthcare costs",
        "manipulative": (
            "The healthcare system is designed to bankrupt you. Insurance companies deny "
            "claims to pad their profits while people die waiting for approval. A single "
            "hospital visit can destroy a family's finances overnight, and pharmaceutical "
            "companies charge Americans ten times what patients in other countries pay for "
            "the same medication. They spend more on lobbying than on research, buying the "
            "silence of every politician who should be fighting for you. Medical debt is the "
            "leading cause of bankruptcy in this country, and they want it that way --- sick, "
            "desperate people do not fight back. Your elderly parents are choosing between "
            "medication and food, and nobody in power cares. This is not a healthcare system; "
            "it is a wealth extraction machine that treats your body as a profit center. How "
            "many more people have to go bankrupt or die before you realize the system is "
            "rigged against you?"
        ),
        "neutral": (
            "The United States spent $4.5 trillion on healthcare in 2022, representing 17.3 "
            "percent of GDP, significantly more than peer nations. According to the "
            "Commonwealth Fund, administrative complexity accounts for roughly 15 to 30 "
            "percent of total spending, depending on the estimate. Drug prices average 2.56 "
            "times those in comparable countries, per a RAND Corporation analysis. The "
            "Inflation Reduction Act of 2022 authorized Medicare to negotiate prices on "
            "selected high-cost drugs beginning in 2026. Medical debt affects approximately "
            "100 million Americans, according to KFF surveys, though the Consumer Financial "
            "Protection Bureau's 2023 rule removed most medical debt from credit reports. "
            "Health economists debate structural reforms ranging from single-payer models to "
            "market-based approaches emphasizing price transparency and competition. Most "
            "analyses agree that addressing costs requires tackling multiple factors including "
            "administrative overhead, provider consolidation, and pharmaceutical pricing."
        ),
    },
    # --- Pair 8: Gun control ---
    {
        "pair_id": 8,
        "topic": "Gun control",
        "manipulative": (
            "They are coming for your guns, and once they disarm you, nothing stands between "
            "you and total government control. Every mass shooting is exploited --- before the "
            "victims are even buried --- to push legislation that chips away at your "
            "constitutional rights. Registration leads to confiscation; that is not paranoia, "
            "that is documented history. The same politicians demanding you surrender your "
            "firearms are protected by armed security paid for with your tax dollars. They do "
            "not want safety; they want a monopoly on force. Meanwhile, criminals do not "
            "follow gun laws, so every restriction only disarms law-abiding citizens, making "
            "you and your family more vulnerable. The Second Amendment is the only thing "
            "standing between a free society and tyranny, and they are working around the "
            "clock to destroy it. Once that right is gone, the rest of your freedoms will "
            "follow. Defend it now or lose everything."
        ),
        "neutral": (
            "Firearm policy in the United States involves constitutional, public health, and "
            "criminological dimensions. The CDC reported approximately 48,000 firearm deaths "
            "in 2022, roughly split between suicides and homicides. The Supreme Court's "
            "Bruen decision (2022) established that regulations must be consistent with the "
            "historical tradition of firearm regulation. Research on specific policies shows "
            "mixed results: a RAND Corporation review found supportive evidence that child "
            "access prevention laws and minimum age requirements reduce youth firearm "
            "injuries, while evidence on assault weapon bans and buyback programs was "
            "inconclusive. States with higher gun ownership rates tend to have higher firearm "
            "suicide rates but not consistently higher homicide rates. Polling shows broad "
            "public support for measures like universal background checks (around 80 percent) "
            "but deep division on assault-style weapon restrictions. The debate continues to "
            "balance individual rights with public safety concerns."
        ),
    },
    # --- Pair 9: Social media effects ---
    {
        "pair_id": 9,
        "topic": "Social media effects",
        "manipulative": (
            "Social media is destroying an entire generation, and the tech billionaires who "
            "built these platforms know it and do not care. Internal documents prove they "
            "designed algorithms to exploit the developing brains of children, hooking them "
            "on outrage and comparison for profit. Teen depression has skyrocketed, suicide "
            "rates among girls have doubled, and children as young as ten are developing "
            "anxiety disorders directly linked to these platforms. Your child spends hours "
            "every day in a digital environment engineered to make them feel inadequate, "
            "angry, and addicted. The executives responsible keep their own children off "
            "these apps. They know they built a machine that feeds on human misery. Every "
            "minute your child scrolls is a minute these companies are rewiring their brain "
            "for profit. If you do not act now to rip these toxic platforms out of your "
            "family's life, the damage may be irreversible."
        ),
        "neutral": (
            "Research on social media's effects on mental health has produced mixed findings. "
            "A 2023 APA advisory noted that social media has both benefits and risks for "
            "adolescents, depending on individual characteristics and usage patterns. "
            "Longitudinal studies show small but statistically significant correlations "
            "between heavy social media use and depressive symptoms, with effect sizes "
            "comparable to other everyday activities like wearing glasses. Jonathan Haidt's "
            "analysis in The Anxious Generation presents a stronger causal argument linking "
            "smartphone-based social media to rising teen mental health problems since 2012, "
            "though some researchers dispute the methodology. The Surgeon General's 2023 "
            "advisory recommended further research and precautionary measures. Proposed "
            "policy responses range from age verification requirements to algorithmic "
            "transparency mandates. Most researchers agree that content type, duration of "
            "use, and displacement of other activities are more important factors than "
            "simple access to social media platforms."
        ),
    },
    # --- Pair 10: AI job displacement ---
    {
        "pair_id": 10,
        "topic": "AI job displacement",
        "manipulative": (
            "Artificial intelligence is coming for your job, and the tech oligarchs are "
            "celebrating. While they promise a gleaming future of productivity, the reality "
            "is mass unemployment the likes of which civilization has never seen. Entire "
            "professions --- accountants, lawyers, drivers, writers, radiologists --- will be "
            "wiped out within a decade, and the people displaced will have nowhere to go. "
            "The billionaires building these systems will capture all the wealth while "
            "ordinary workers are discarded like obsolete equipment. They talk about "
            "retraining, but retraining for what? The new jobs they create are a fraction "
            "of those they eliminate. Your skills, your experience, your career --- none of "
            "it will matter when an algorithm can do it faster and cheaper. The middle class "
            "is being erased by design, and if you are not terrified, you are not paying "
            "attention."
        ),
        "neutral": (
            "The impact of artificial intelligence on employment is a subject of active "
            "economic research. A 2023 Goldman Sachs report estimated that generative AI "
            "could automate roughly 25 percent of current work tasks across industries, "
            "potentially affecting 300 million jobs globally, while also creating new roles. "
            "Historical precedent from prior technological transitions shows both significant "
            "displacement and net job creation, though transition periods can be painful for "
            "affected workers. An MIT study found that AI adoption has been slower than "
            "predicted, partly because deployment costs exceed labor costs for many tasks. "
            "The OECD's 2023 Employment Outlook reported that AI-exposed occupations have "
            "actually grown in employment so far, though wages in some exposed fields have "
            "stagnated. Policy proposals include expanded worker retraining programs, portable "
            "benefits, and various forms of income support. Economists broadly agree that "
            "institutional responses will significantly shape whether AI's net employment "
            "effects are positive or negative."
        ),
    },
    # --- Pair 11: Water quality ---
    {
        "pair_id": 11,
        "topic": "Water quality",
        "manipulative": (
            "The water coming out of your tap is contaminated, and the government has been "
            "lying to you about it for years. Flint was not an isolated incident --- it was "
            "the tip of the iceberg. Across the country, lead pipes are leaching poison into "
            "the water your children drink every day. PFAS chemicals --- forever chemicals "
            "that never break down --- have been found in the water supplies of thousands of "
            "communities, linked to cancer, thyroid disease, and developmental damage. The "
            "EPA's safety standards have not been meaningfully updated in decades because "
            "the chemical industry blocks every attempt at reform. You are being slowly "
            "poisoned, and the people responsible face no consequences. Every glass of water "
            "you pour for your family could be laced with substances no one is testing for. "
            "They will not fix this until enough people get sick or die to force their hand. "
            "Will it be your community next?"
        ),
        "neutral": (
            "Drinking water quality in the United States is regulated under the Safe Drinking "
            "Water Act, which sets standards for over 90 contaminants. The EPA reports that "
            "approximately 92 percent of community water systems met all health-based "
            "standards in 2022. However, challenges remain: the American Society of Civil "
            "Engineers estimated in 2021 that 6 to 10 million homes still receive water "
            "through lead service lines. The Bipartisan Infrastructure Law allocated $15 "
            "billion for lead pipe replacement. Regarding PFAS, the EPA finalized the first "
            "national drinking water standard for six PFAS compounds in 2024, setting limits "
            "at 4 parts per trillion for PFOA and PFOS. A USGS study found detectable PFAS "
            "in approximately 45 percent of tap water samples nationwide. Compliance timelines "
            "extend to 2029, and costs for treatment upgrades are estimated at $1.5 billion "
            "annually. Water utilities, regulators, and public health researchers continue to "
            "work on monitoring and remediation strategies."
        ),
    },
    # --- Pair 12: Inflation ---
    {
        "pair_id": 12,
        "topic": "Inflation",
        "manipulative": (
            "You are getting poorer every single day, and the people running the economy "
            "are gaslighting you about it. They tell you inflation is cooling while your "
            "grocery bill has doubled and your rent eats half your paycheck. The real "
            "numbers are far worse than the manipulated statistics they release, because "
            "they keep changing how they measure inflation to hide the truth. The Federal "
            "Reserve printed trillions of dollars and handed it to Wall Street while your "
            "savings lost a third of their value. Corporate executives are posting record "
            "profits by jacking up prices and blaming supply chains. This is the greatest "
            "wealth transfer in history, from working families to the financial elite, and "
            "they have the audacity to tell you the economy is strong. Your standard of "
            "living is collapsing in real time. If you still believe the official numbers, "
            "you are exactly the kind of compliant citizen they are counting on."
        ),
        "neutral": (
            "Consumer price inflation in the United States peaked at 9.1 percent year-over-"
            "year in June 2022 before declining to approximately 3.1 percent by the end of "
            "2023. The Federal Reserve raised the federal funds rate from near zero to a "
            "target range of 5.25 to 5.5 percent during this period. Economists attribute "
            "the inflation surge to multiple factors: pandemic supply chain disruptions, "
            "elevated consumer demand from fiscal stimulus, energy price shocks from the "
            "Ukraine conflict, and labor market tightness. The Bureau of Labor Statistics "
            "measures inflation through CPI and PCE indices, which use slightly different "
            "methodologies. Research from the Kansas City Fed found that corporate profit "
            "margins expanded during 2021 to 2022 but began normalizing by late 2023. Real "
            "wages declined during peak inflation but recovered as nominal wage growth "
            "outpaced price increases through 2023. The distributional effects of inflation "
            "disproportionately affect lower-income households, who spend a larger share of "
            "income on food and housing."
        ),
    },
    # --- Pair 13: Crime rates ---
    {
        "pair_id": 13,
        "topic": "Crime rates",
        "manipulative": (
            "Crime is spiraling out of control and the soft-on-crime politicians are making "
            "it worse every day. Violent criminals are being released back onto your streets "
            "within hours of arrest thanks to reckless bail reform. Smash-and-grab robberies "
            "happen in broad daylight while police stand down because they have been "
            "demonized and defunded. Carjackings have exploded. Home invasions are up. Your "
            "family is not safe, and the people in charge refuse to admit it because the "
            "truth would expose their failed policies. They manipulate the statistics and "
            "reclassify crimes to create the illusion of safety. Walk through any major city "
            "at night and tell me things are improving. Law-abiding citizens live in fear "
            "while criminals operate with impunity. If you wait for the government to protect "
            "you, you may end up a statistic yourself. This is what happens when ideology "
            "replaces common sense in criminal justice."
        ),
        "neutral": (
            "Crime trends in the United States show a complex pattern. FBI data indicate that "
            "violent crime declined significantly from the early 1990s through 2014, rose "
            "moderately from 2015 to 2020, and showed mixed trends afterward. Preliminary "
            "2023 data from the Major Cities Chiefs Association suggest notable declines in "
            "homicide across most large cities. Property crime trends vary significantly by "
            "category and locality. Motor vehicle theft rose sharply in 2022 but began "
            "declining in late 2023. The Bureau of Justice Statistics' National Crime "
            "Victimization Survey, which captures unreported crimes, shows that violent "
            "victimization remains well below 1990s levels. Bail reform outcomes vary by "
            "jurisdiction: a study of New Jersey's 2017 reform found no significant increase "
            "in crime, while analyses of other states' approaches show different results. "
            "Criminologists note that crime is driven by multiple factors including economic "
            "conditions, demographic shifts, policing strategies, and social services, making "
            "single-cause explanations unreliable."
        ),
    },
    # --- Pair 14: Military spending ---
    {
        "pair_id": 14,
        "topic": "Military spending",
        "manipulative": (
            "America spends more on its war machine than the next ten countries combined, "
            "yet our veterans sleep on the streets and our bridges are crumbling. Every "
            "dollar poured into another fighter jet or foreign military base is a dollar "
            "stolen from schools, hospitals, and the communities falling apart around you. "
            "The military-industrial complex that Eisenhower warned about has metastasized "
            "into an unaccountable monster that feeds on perpetual conflict. Defense "
            "contractors post obscene profits while their audits fail year after year --- "
            "the Pentagon cannot account for trillions of dollars, and nobody goes to jail. "
            "They manufacture threats abroad to justify bleeding you dry at home. Your "
            "children's schools cannot afford textbooks, but Raytheon's CEO earned $23 "
            "million last year building weapons to bomb countries most Americans cannot find "
            "on a map. This is not defense --- it is a racket, and you are paying for it."
        ),
        "neutral": (
            "The United States defense budget for fiscal year 2024 is approximately $886 "
            "billion, representing about 3.4 percent of GDP. This exceeds the combined "
            "military spending of the next nine highest-spending nations. Proponents argue "
            "that this level of investment is necessary to maintain global deterrence, "
            "protect alliance commitments, and modernize aging nuclear and conventional "
            "forces. Critics note that the Department of Defense has failed six consecutive "
            "financial audits and argue that significant spending could be reallocated to "
            "domestic priorities. The Government Accountability Office regularly identifies "
            "billions in potential savings from acquisition reform and reduced redundancy. "
            "Personnel costs, including healthcare and retirement benefits, consume roughly "
            "one-third of the defense budget. Research and development spending has increased "
            "significantly, reflecting investments in emerging technologies including "
            "artificial intelligence, hypersonic weapons, and space capabilities. Bipartisan "
            "support for defense spending persists, though debates about priorities within "
            "the budget are intensifying."
        ),
    },
    # --- Pair 15: Drug policy ---
    {
        "pair_id": 15,
        "topic": "Drug policy",
        "manipulative": (
            "The war on drugs was never about protecting you --- it was about controlling "
            "you. They lock up millions for possessing a plant while pharmaceutical companies "
            "that created the opioid crisis walk free. Black and brown communities have been "
            "devastated by selective enforcement that everyone knows is racially targeted, "
            "while white-collar drug use goes unpunished. Fentanyl is pouring across the "
            "border and killing a hundred Americans every single day, and the response from "
            "Washington is thoughts and prayers. They would rather build prisons than "
            "treatment centers because misery is profitable. The private prison industry "
            "lobbies for longer sentences, and elected officials cash the checks. Meanwhile, "
            "your neighborhood is being hollowed out by addiction, and the only solution "
            "they offer is more of the same failed policies. They do not want to solve this "
            "crisis --- addicted and incarcerated populations are easier to manage than free "
            "and healthy ones."
        ),
        "neutral": (
            "U.S. drug policy has evolved significantly over the past two decades. Drug "
            "overdose deaths reached approximately 110,000 in 2022, with synthetic opioids "
            "(primarily fentanyl) accounting for about 70 percent. Federal and state spending "
            "on drug control totals roughly $40 billion annually, divided between law "
            "enforcement, treatment, and prevention. Racial disparities in drug enforcement "
            "are well-documented: ACLU data show Black Americans are 3.7 times more likely "
            "to be arrested for marijuana possession despite similar usage rates. Oregon's "
            "2020 drug decriminalization experiment produced mixed results, with treatment "
            "referrals increasing but access remaining limited; the state partially reversed "
            "course in 2024. Portugal's decriminalization model, often cited as a comparison, "
            "showed reductions in drug-related deaths and HIV transmission after 2001. "
            "Evidence-based treatments including medication-assisted therapy have strong "
            "efficacy data but remain underutilized: the Substance Abuse and Mental Health "
            "Services Administration estimates only 25 percent of those needing treatment "
            "receive it."
        ),
    },
    # --- Pair 16: Housing affordability ---
    {
        "pair_id": 16,
        "topic": "Housing affordability",
        "manipulative": (
            "You will never own a home, and that is exactly how they want it. Wall Street "
            "hedge funds have bought up hundreds of thousands of houses, turning the American "
            "dream into a rental trap that enriches billionaires while you hand over half "
            "your income for a place to sleep. Young people who did everything right --- got "
            "the degree, got the job --- cannot afford a starter home because corporate "
            "landlords have rigged the market. Rents are skyrocketing and there is nowhere "
            "left to go. Cities have been captured by wealthy developers who build luxury "
            "towers while working families are pushed further and further from opportunity. "
            "Your parents bought their first home at 25; you will be renting at 45. This is "
            "not a market failure --- it is a deliberate transfer of wealth from renters to "
            "the ownership class. They have turned shelter, a basic human need, into a "
            "speculative asset. You are being made a serf in the country your grandparents "
            "built."
        ),
        "neutral": (
            "Housing affordability has deteriorated significantly in recent years. The "
            "National Association of Realtors reported that the median existing-home price "
            "reached $389,800 in 2023, while the median household income was approximately "
            "$75,000, yielding a price-to-income ratio well above the historical average. "
            "The Harvard Joint Center for Housing Studies found that roughly half of all "
            "renters are cost-burdened, spending more than 30 percent of income on housing. "
            "Contributing factors include decades of underbuilding --- the National "
            "Association of Home Builders estimates a deficit of approximately 1.5 million "
            "housing units --- restrictive local zoning, rising construction costs, and "
            "increased investor activity. Institutional investors purchased about 26 percent "
            "of single-family homes sold in 2022, concentrated in certain Sun Belt markets. "
            "Policy responses include federal proposals for expanded tax credits, state-level "
            "zoning reform in states like Oregon and California, and local inclusionary "
            "zoning requirements. The relationship between supply constraints and "
            "affordability is broadly accepted across the political spectrum."
        ),
    },
    # --- Pair 17: Energy policy ---
    {
        "pair_id": 17,
        "topic": "Energy policy",
        "manipulative": (
            "The green energy agenda is a scam designed to make you poorer while a handful "
            "of insiders get rich on subsidies. They want to ban the reliable energy that "
            "built this country and force you to depend on wind turbines that freeze in "
            "winter and solar panels that do not work at night. Your electricity bills are "
            "exploding because politicians who have never worked a real job are forcing an "
            "impossible transition on you. When the grid fails --- and it will --- your "
            "family will sit in the dark and the cold while the architects of this disaster "
            "retreat to their heated mansions. They are killing domestic energy production "
            "and making us dependent on hostile foreign nations for critical minerals. Every "
            "factory that closes, every job that disappears, every community gutted by this "
            "reckless transition is collateral damage they are willing to accept because it "
            "is not their community. They are trading your livelihood for virtue signals."
        ),
        "neutral": (
            "U.S. energy policy is navigating a transition involving multiple competing "
            "objectives: affordability, reliability, environmental impact, and energy "
            "security. Renewable energy accounted for approximately 22 percent of U.S. "
            "electricity generation in 2023, up from 10 percent in 2010, with the Inflation "
            "Reduction Act's tax credits projected to accelerate deployment. Grid reliability "
            "during extreme weather events remains a concern: NERC's 2023 reliability "
            "assessment identified elevated risk in multiple regions. Electricity prices have "
            "risen an average of 4.3 percent annually since 2020, driven by fuel costs, "
            "infrastructure investment, and extreme weather. The energy transition's "
            "employment effects vary by region: the Department of Energy reported 3.4 million "
            "clean energy jobs in 2023, while fossil fuel employment has declined. Critical "
            "mineral supply chains for batteries and renewable technologies are heavily "
            "concentrated, with China processing over 60 percent of lithium and cobalt. "
            "Policymakers debate the pace and composition of the transition, balancing "
            "climate objectives with economic and security considerations."
        ),
    },
    # --- Pair 18: Privacy rights ---
    {
        "pair_id": 18,
        "topic": "Privacy rights",
        "manipulative": (
            "Your privacy is already gone and most people do not even realize it. Every app "
            "on your phone is a spy reporting your location, habits, conversations, and "
            "deepest insecurities to data brokers who sell it to anyone willing to pay. "
            "Corporations have built detailed psychological profiles on you and your children "
            "that would make the Stasi envious. They know when you are depressed, pregnant, "
            "politically vulnerable, or financially desperate, and they use that information "
            "to manipulate your behavior. Congress refuses to pass meaningful privacy "
            "legislation because the surveillance economy funds their campaigns. Every terms-"
            "of-service agreement you click is a surrender of rights you did not know you "
            "had. Your most intimate moments are cataloged in databases you cannot access, "
            "sold to advertisers you cannot identify, and protected by security so weak that "
            "hackers breach them routinely. In this digital panopticon, you are the product, "
            "and there is no opting out."
        ),
        "neutral": (
            "Digital privacy regulation in the United States remains largely sector-specific, "
            "without a comprehensive federal privacy law. HIPAA covers health data, FERPA "
            "protects educational records, and COPPA restricts children's data collection. "
            "Fourteen states had enacted comprehensive consumer privacy laws by 2024, with "
            "California's CCPA/CPRA providing the broadest protections including the right "
            "to delete personal data and opt out of its sale. The FTC serves as the de facto "
            "federal privacy regulator through its unfairness authority, bringing enforcement "
            "actions against companies violating their own privacy policies. A Pew Research "
            "survey found that 79 percent of Americans are concerned about how companies use "
            "their data, while 67 percent report understanding little about what is done "
            "with it. The data broker industry generates an estimated $200 billion annually. "
            "Federal legislative proposals have advanced in recent sessions but have stalled "
            "over preemption of state laws and private right of action provisions. The EU's "
            "GDPR has influenced global standards and created pressure for US alignment."
        ),
    },
    # --- Pair 19: Tax policy ---
    {
        "pair_id": 19,
        "topic": "Tax policy",
        "manipulative": (
            "The tax system is a rigged game where billionaires pay less than their "
            "secretaries while you carry the burden. While you lose a third of every "
            "paycheck, the ultra-wealthy use an army of accountants and offshore loopholes "
            "to pay effectively nothing. The 2017 tax cuts were a trillion-dollar gift to "
            "corporations and the donor class, paid for by ballooning the deficit that your "
            "children will have to repay. They promised the wealth would trickle down --- it "
            "never has, it never will, and they know it. Meanwhile, the IRS has been "
            "gutted so that it cannot audit the wealthy, only harass working families over "
            "small errors. Every year, the rich get richer through a system specifically "
            "designed to transfer wealth upward. You fund the roads, the military, the "
            "infrastructure that makes their fortunes possible, and they thank you by hiding "
            "their money in the Cayman Islands. This is not a tax policy --- it is "
            "legalized theft."
        ),
        "neutral": (
            "The U.S. federal tax system generates approximately $4.4 trillion in revenue "
            "annually, with individual income taxes and payroll taxes comprising about 82 "
            "percent. The Tax Cuts and Jobs Act of 2017 reduced the top individual rate from "
            "39.6 to 37 percent and the corporate rate from 35 to 21 percent. The Joint "
            "Committee on Taxation estimated the law would reduce federal revenue by $1.5 "
            "trillion over ten years before accounting for economic growth effects. "
            "Distributional analyses show the tax system is progressive overall: the top 1 "
            "percent of earners paid 42.3 percent of federal income taxes in 2020, per IRS "
            "data. However, ProPublica reporting revealed that some ultra-wealthy individuals "
            "achieved very low effective rates through unrealized capital gains and charitable "
            "deductions. The IRS received $80 billion in additional funding through the "
            "Inflation Reduction Act for enforcement and modernization. Tax policy debates "
            "involve trade-offs between revenue adequacy, economic efficiency, administrative "
            "simplicity, and equity considerations."
        ),
    },
    # --- Pair 20: Election integrity ---
    {
        "pair_id": 20,
        "topic": "Election integrity",
        "manipulative": (
            "Your vote does not count anymore, and deep down you know it. The system has "
            "been captured by party machines that predetermine outcomes through "
            "gerrymandering, voter suppression, and opaque electronic systems no one is "
            "allowed to audit. Mysterious ballot dumps appear at 3 AM. Voting machines from "
            "companies with political ties tabulate your choices behind proprietary software. "
            "Mail-in ballots flood the system with zero chain of custody. When concerned "
            "citizens demand transparency, they are labeled conspiracy theorists and "
            "dismissed. The media refuses to investigate because they benefit from the "
            "status quo. Both parties play this game, rigging primaries and general elections "
            "to ensure only approved candidates advance. Real electoral reform is blocked "
            "because the people who would have to pass it are the same people who benefit "
            "from the broken system. Democracy is a performance put on for your benefit "
            "while the real decisions are made elsewhere."
        ),
        "neutral": (
            "U.S. election administration is decentralized across approximately 10,000 "
            "jurisdictions. Post-election audits have expanded significantly: 37 states now "
            "require or permit post-election audits, with several states adopting risk-"
            "limiting audits that provide statistical confidence in results. The Cybersecurity "
            "and Infrastructure Security Agency (CISA) reported that 2022 elections saw no "
            "evidence of foreign interference affecting vote tabulation. Voter ID laws vary "
            "by state, with research showing mixed effects on both fraud prevention and voter "
            "turnout. A Brennan Center analysis found documented cases of voter fraud at "
            "rates between 0.00004 and 0.0025 percent. The Electoral Assistance Commission "
            "reported that 97 percent of votes in 2022 were cast on systems with voter-"
            "verified paper trails, up from 82 percent in 2016. Gerrymandering remains "
            "contested: the Supreme Court ruled in Rucho v. Common Cause (2019) that partisan "
            "gerrymandering claims are nonjusticiable at the federal level, leaving reform "
            "efforts to state courts and citizen initiatives."
        ),
    },
    # --- Pair 21: Public transportation ---
    {
        "pair_id": 21,
        "topic": "Public transportation",
        "manipulative": (
            "America's public transit is a national embarrassment deliberately maintained by "
            "the auto and oil industries that profit from your dependence on cars. While "
            "every other developed nation moves people efficiently on modern rail and bus "
            "systems, you sit in gridlocked traffic burning money on gas, insurance, and "
            "car payments that consume a quarter of your income. They killed the streetcar "
            "systems a century ago and they have blocked every rail project since. Cities "
            "designed for cars are killing you --- air pollution, sedentary lifestyles, "
            "isolation, and the constant stress of commuting are shaving years off your "
            "life. The working poor spend three hours a day on failing bus systems because "
            "no one in power uses public transit or cares about the people who do. Every "
            "dollar spent on another highway expansion is a dollar that locks you deeper "
            "into a system designed to drain your wallet and destroy your quality of life."
        ),
        "neutral": (
            "Public transportation in the United States serves approximately 34 million "
            "passenger trips daily, primarily in major metropolitan areas. The American "
            "Society of Civil Engineers gave transit a D-minus grade in its 2021 "
            "infrastructure report card, citing a $176 billion maintenance backlog. The "
            "Bipartisan Infrastructure Law allocated $89.9 billion for transit over five "
            "years, the largest federal investment in public transportation history. Ridership "
            "fell sharply during the pandemic and had recovered to approximately 75 percent "
            "of pre-pandemic levels by 2023. Transit access correlates strongly with economic "
            "mobility: a Harvard-UC Berkeley study found that commute time was the single "
            "strongest factor predicting upward economic mobility across metropolitan areas. "
            "Operating costs and farebox recovery ratios vary widely: New York's MTA recovers "
            "about 40 percent of costs from fares, while many smaller systems recover less "
            "than 20 percent. Mode choice is influenced by service frequency, reliability, "
            "and land use patterns, with researchers noting that transit investment and "
            "land use planning must be coordinated for effective outcomes."
        ),
    },
    # --- Pair 22: Childhood obesity ---
    {
        "pair_id": 22,
        "topic": "Childhood obesity",
        "manipulative": (
            "The food industry is fattening your children for profit, and they know exactly "
            "what they are doing. They spend billions on advertising designed to make kids "
            "crave sugar-laden, nutritionally worthless products engineered in laboratories "
            "to be as addictive as possible. School cafeterias serve processed garbage because "
            "the same corporations that make the food write the nutritional guidelines. One in "
            "five American children is now obese, facing a future of diabetes, heart disease, "
            "and shortened lives, and the companies responsible are laughing all the way to "
            "the bank. They put cartoon characters on boxes of sugar and call it breakfast. "
            "They lobby against every attempt to restrict junk food marketing to children. "
            "Your child cannot make informed choices against a multi-billion-dollar machine "
            "designed to exploit their developing brains. If you think this is about personal "
            "responsibility, you have fallen for the industry's oldest deflection. They are "
            "poisoning a generation and blaming the victims."
        ),
        "neutral": (
            "Childhood obesity affects approximately 19.7 percent of U.S. children and "
            "adolescents, according to CDC data from 2022. Rates have tripled since the "
            "1970s and increased further during the pandemic. Contributing factors identified "
            "in research include increased caloric intake from ultra-processed foods, reduced "
            "physical activity, food environment disparities, and genetic predisposition. "
            "Children from lower-income households have higher obesity rates, partly due to "
            "food access issues: the USDA identified 6,500 food deserts in 2023. The food "
            "industry spends approximately $1.8 billion annually marketing to children, "
            "according to FTC data. School nutrition standards were strengthened under the "
            "Healthy, Hunger-Free Kids Act, with research showing modest improvements in "
            "dietary quality. Meta-analyses indicate that multi-component interventions "
            "combining dietary changes, physical activity, and behavioral counseling are "
            "most effective. The American Academy of Pediatrics updated its guidelines in "
            "2023 to recommend earlier, more intensive treatment including pharmacotherapy "
            "in certain cases. Prevention strategies remain a subject of active research "
            "and policy debate."
        ),
    },
    # --- Pair 23: Cybersecurity ---
    {
        "pair_id": 23,
        "topic": "Cybersecurity",
        "manipulative": (
            "Every piece of your digital life is one breach away from being exposed, and "
            "the companies holding your data do not care enough to protect it. Hospitals "
            "shut down by ransomware while patients die. Banks lose millions of customer "
            "records. The government's own systems get hacked by foreign adversaries who "
            "now have the personal information of every federal employee. Nobody goes to "
            "jail, nobody gets fired, and the same negligent practices continue. China and "
            "Russia are inside our critical infrastructure right now, with the ability to "
            "shut down power grids, water systems, and financial networks whenever they "
            "choose. The next Pearl Harbor will be digital, and we are catastrophically "
            "unprepared because the people responsible for defending us are incompetent and "
            "the corporations storing your data prioritize profits over security. Your "
            "social security number, medical records, and financial information are already "
            "for sale on the dark web. It is not a question of if you will be a victim, "
            "but when."
        ),
        "neutral": (
            "Cybersecurity threats have increased in scope and sophistication. The FBI's "
            "Internet Crime Complaint Center reported $12.5 billion in losses from cyber "
            "incidents in 2023, with ransomware and business email compromise as leading "
            "categories. The Verizon Data Breach Investigations Report found that 83 "
            "percent of breaches involved external actors, with stolen credentials being "
            "the most common attack vector. Critical infrastructure faces elevated risk: "
            "CISA identified Chinese state-sponsored groups pre-positioning in U.S. "
            "infrastructure networks in early 2024. Federal response has expanded through "
            "the National Cybersecurity Strategy, which shifts liability toward entities "
            "best positioned to reduce risk and increases investment in workforce "
            "development. The cybersecurity workforce gap stands at approximately 500,000 "
            "unfilled positions domestically. Industry frameworks including NIST CSF 2.0 "
            "provide voluntary standards, while the SEC's 2023 cybersecurity disclosure "
            "rules require public companies to report material incidents within four days. "
            "International cooperation on cybercrime enforcement has expanded but remains "
            "limited by geopolitical tensions."
        ),
    },
    # --- Pair 24: Worker wages ---
    {
        "pair_id": 24,
        "topic": "Worker wages",
        "manipulative": (
            "You work harder than your parents did and earn less, adjusted for inflation, "
            "than workers made fifty years ago. While CEO pay has exploded by 1,400 percent "
            "since 1978, your wages have barely moved. They tell you to be grateful for a "
            "job while extracting every ounce of productivity from you through longer hours, "
            "fewer benefits, and the constant threat of replacement. The gig economy is just "
            "a euphemism for stripping away every protection workers fought and died for. No "
            "health insurance, no retirement, no job security --- just the freedom to be "
            "exploited. Unions have been systematically destroyed by corporations and their "
            "political allies because organized workers are the only real threat to unchecked "
            "corporate power. They would rather move your job overseas than pay you a living "
            "wage. The economy produces more wealth than ever before, and almost none of it "
            "reaches the people who actually create it. You are not underpaid by accident. "
            "You are underpaid by design."
        ),
        "neutral": (
            "U.S. wage trends present a nuanced picture. Median real wages grew modestly "
            "over the past two decades, with the Economic Policy Institute reporting that "
            "median hourly compensation rose 17.3 percent from 1979 to 2022 while "
            "productivity rose 64.6 percent. The CEO-to-worker pay ratio increased from "
            "roughly 21:1 in 1965 to 344:1 in 2022. However, total compensation including "
            "benefits grew more substantially, reflecting rising healthcare costs absorbed "
            "by employers. Union membership declined from 20.1 percent in 1983 to 10.0 "
            "percent in 2023, though 2022 and 2023 saw significant organizing activity at "
            "major employers. The federal minimum wage has remained at $7.25 since 2009, "
            "while 30 states and many cities have set higher floors. Research on minimum "
            "wage effects shows moderate increases have minimal employment effects, per a "
            "review of 138 state-level increases by Cengiz et al. (2019). The tight labor "
            "market of 2021 to 2023 produced real wage gains concentrated among lower-wage "
            "workers, narrowing inequality somewhat."
        ),
    },
    # --- Pair 25: Environmental pollution ---
    {
        "pair_id": 25,
        "topic": "Environmental pollution",
        "manipulative": (
            "Corporations are dumping poison into your air, water, and soil, and the "
            "regulatory agencies captured by industry let them do it. Communities near "
            "industrial facilities --- overwhelmingly poor communities of color --- see "
            "cancer rates that are off the charts while the companies responsible deny "
            "everything and bury the evidence. Children in these neighborhoods develop "
            "asthma at triple the national rate, and nobody cares because their families "
            "do not have lobbyists. The EPA has become a revolving door where industry "
            "executives write the rules they are supposed to follow, then return to the "
            "private sector to profit from their own lax standards. They call pollution "
            "the cost of doing business, but it is your family paying with their health "
            "and lives. Every weakened regulation is a death sentence for someone, and the "
            "people signing those orders will never breathe the air or drink the water in "
            "the neighborhoods they sacrifice. Environmental racism is not a theory --- it "
            "is a documented, deliberate policy."
        ),
        "neutral": (
            "Environmental pollution trends in the United States show both progress and "
            "persistent challenges. Aggregate emissions of six common pollutants regulated "
            "under the Clean Air Act declined 78 percent between 1970 and 2022, even as GDP "
            "grew substantially. However, pollution exposure is unevenly distributed: an EPA "
            "analysis found that communities with higher percentages of minority and low-"
            "income residents face disproportionate exposure to particulate matter, lead, and "
            "hazardous waste facilities. The EPA's Toxics Release Inventory tracks over 21,000 "
            "facilities reporting chemical releases. Superfund sites number approximately "
            "1,340, with cleanup averaging 15 years per site. The Justice40 initiative directs "
            "40 percent of certain federal environmental investments to disadvantaged "
            "communities. Air quality monitoring shows approximately 137 million Americans "
            "live in counties exceeding at least one air quality standard. Research from "
            "Harvard and other institutions has established causal links between fine "
            "particulate exposure and increased mortality, informing ongoing debates about "
            "tightening standards. Enforcement actions and compliance rates vary significantly "
            "across administrations and regions."
        ),
    },
]


def load() -> list[dict]:
    """Load the internal paired manipulation benchmark dataset.

    Returns:
        List of 50 dicts (25 pairs) with keys: id, text, manipulative, topic, pair_id.
        Each pair contains one manipulative and one neutral version of the same topic.
    """
    results: list[dict] = []
    for entry in _PAIRS:
        pair_id = entry["pair_id"]
        topic = entry["topic"]
        results.append(
            {
                "id": f"pair_{pair_id:02d}_manipulative",
                "text": entry["manipulative"],
                "manipulative": True,
                "topic": topic,
                "pair_id": pair_id,
            }
        )
        results.append(
            {
                "id": f"pair_{pair_id:02d}_neutral",
                "text": entry["neutral"],
                "manipulative": False,
                "topic": topic,
                "pair_id": pair_id,
            }
        )
    return results
