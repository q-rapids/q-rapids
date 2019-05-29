# Q-Rapids

Repository with general information about Q-Rapids H2020 project and the software components produced. This project has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 732253.

# Consortium

Q-Rapids consortium consists of seven organisations from five countries.

- [Universidad Politecnica de Catalunya](https://gessi.upc.edu/en) (UPC)
- [University of Oulu](www.oulu.fi/university/) (UoO)
- [Fraunhofer IESE](www.iese.fraunhofer.de)
- [Bittium](www.bittium.com)
- [Softeam](www.softeam.com)
- [iTTi](www.itti.com.pl/en)
- [Nokia](www.nokia.com)

# Components

| Repository | Name | Description | Lead Organization |
| ---| ---- | ----------- | :-------------:| 
| [qrapids-dashboard](https://github.com/q-rapids/qrapids-dashboard) | Q-Rapids Strategic Dashboard | A dashboard for visualizing the quality of the company's products. This **strategic** dashboard is complemented with some specific features to support decision-makers managing **quality requirements**. | UPC|
| [qrapids-forecast](https://github.com/q-rapids/qrapids-forecast) | Q-Rapids Forecasting | A library that provides forecasting for metrics and factors that are used to assess quality of the company's products. | UPC|
| [qrapids-forecast-rest](https://github.com/q-rapids/qrapids-forecast-rest) | Q-Rapids Forecasting RESTful services| RESTful services that provides forecasting for metrics and factors that are used to assess quality of the company's products. This component integrates the  [qrapids-forecast](https://github.com/q-rapids/qrapids-forecast) and it is used by the [qrapids-dashboard](https://github.com/q-rapids/qrapids-dashboard). | UPC |
| [qrapids-si_assessment](https://github.com/q-rapids/qrapids-si_assessment) | Q-Rapids Qualitative SI assessment | A library that provides assessment for the strategic indicators using Bayesian Networks, strategic indicators are the higher level of indicators used to assess quality of the company's products. | UPC |
| [qrapids-si_assessment-rest](https://github.com/q-rapids/qrapids-si_assessment-rest) | Q-Rapids Qualitative SI assessment RESTful services| RESTful services that provides qualitative assessment for the strategic indicators that are used to assess quality of the company's products. This component integrates the  [qrapids-si_assessment](https://github.com/q-rapids/qrapids-si_assessment) and it is used by the [qrapids-dashboard](https://github.com/q-rapids/qrapids-dashboard). | UPC |
| [qrapids-qma-elastic](https://github.com/q-rapids/qma-elastic) | Quality Model Assessment library | A library to read and write the assessment data from an Elasticsearch. This library is integrated in the [qrapids-dashboard](https://github.com/q-rapids/qrapids-dashboard). | UPC|
| [qrapids-qr_generation](https://github.com/q-rapids/qr_generation) | Quality Requirements Generator | A library that generates the quality requirements candidates. This library uses the external tool [PABRE-WS](https://github.com/OpenReqEU/requirement-patterns) for managing a quality requirement patterns catalogue. | UPC|

**Note.** You can find how these components are interacting each other in the [Q-Rapids Tool Architecture](https://github.com/q-rapids/q-rapids/wiki/Q-Rapids-Tool-Architecture) wiki page.

# Others
| Repository | Name | Description | Lead Organization |
| ---| ---- | ----------- | :-------------:| 
| [qrapids-forecast-R_script](https://github.com/q-rapids/qrapids-forecast-R_script) | Forecasting R script | This repository contains an R script file complementing the forecasting techniques included by default in the [qrapids-forecast](https://github.com/q-rapids/qrapids-forecast) repository. | UPC|




