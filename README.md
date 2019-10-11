# <img src="https://github.com/q-rapids/q-rapids/wiki/images/logos/qrapids_logo_color_100x100_noname.png" width="60">Q-Rapids: Quality-aware rapid software development

Repository with general information about Q-Rapids H2020 project and the software components produced. The aim of the project is produce an evidence-based, data-driven quality-aware rapid software development method where quality requirements are incrementally elicited, refined and improved based on data gathered from software repositories, management tools, system usage and quality of service. This data is analysed and aggregated into quality-related strategic indicators which are presented in a highly informative dashboard.

![](https://github.com/q-rapids/q-rapids/wiki/images/qrapids_framework.png)

This project has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 732253.

# Documentation

You can find all the information you need in this repository [Wiki](https://github.com/q-rapids/q-rapids/wiki).

# Consortium

Q-Rapids consortium consists of seven organisations from five countries.

- [Universidad Politecnica de Catalunya](https://gessi.upc.edu/en) (UPC)
- [University of Oulu](https://www.oulu.fi/university/) (UoO)
- [Fraunhofer IESE](https://www.iese.fraunhofer.de/)
- [Bittium](https://www.bittium.com/)
- [Softeam](https://www.softeamgroup.fr/en)
- [iTTi](https://www.itti.com.pl/en/)
- [Nokia](https://www.nokia.com/es_int/)

# Components

| Repository | Name | Description | Lead Organization |
| ---| ---- | ----------- | :-------------:| 
| [qrapids-dashboard](https://github.com/q-rapids/qrapids-dashboard) | Q-Rapids Strategic Dashboard | A dashboard for visualizing the quality of the company's products. This **strategic** dashboard is complemented with some specific features to support decision-makers managing **quality requirements**. | UPC|
| [qrapids-forecast](https://github.com/q-rapids/qrapids-forecast) | Q-Rapids Forecasting | A library that provides forecasting for metrics and factors that are used to assess quality of the company's products. | UPC|
| [qrapids-forecast-rest](https://github.com/q-rapids/qrapids-forecast-rest) | Q-Rapids Forecasting RESTful services| RESTful services that provides forecasting for metrics and factors that are used to assess quality of the company's products. This component integrates the  [qrapids-forecast](https://github.com/q-rapids/qrapids-forecast) and it is used by the [qrapids-dashboard](https://github.com/q-rapids/qrapids-dashboard). | UPC |
| [qrapids-forecast-R_script](https://github.com/q-rapids/qrapids-forecast-R_script) | Forecasting R script | This repository contains an R script file complementing the forecasting techniques included by default in the [qrapids-forecast](https://github.com/q-rapids/qrapids-forecast) repository. | UPC|
| [qrapids-si_assessment](https://github.com/q-rapids/qrapids-si_assessment) | Q-Rapids Qualitative SI assessment | A library that provides assessment for the strategic indicators using Bayesian Networks, strategic indicators are the higher level of indicators used to assess quality of the company's products. | UPC |
| [qrapids-si_assessment-rest](https://github.com/q-rapids/qrapids-si_assessment-rest) | Q-Rapids Qualitative SI assessment RESTful services| RESTful services that provides qualitative assessment for the strategic indicators that are used to assess quality of the company's products. This component integrates the  [qrapids-si_assessment](https://github.com/q-rapids/qrapids-si_assessment) and it is used by the [qrapids-dashboard](https://github.com/q-rapids/qrapids-dashboard). | UPC |
| [qrapids-qma-elastic](https://github.com/q-rapids/qrapids-qma-elastic) | Quality Model Assessment | A library to read and write the assessment data from an Elasticsearch. This library is integrated in the [qrapids-dashboard](https://github.com/q-rapids/qrapids-dashboard). | UPC|
| [qrapids-qr_generation](https://github.com/q-rapids/qr_generation) | Quality Requirements Generator | A library that generates the quality requirements candidates. This library uses the external tool [PABRE-WS](https://github.com/OpenReqEU/requirement-patterns) for managing a quality requirement patterns catalogue. | UPC|
| [qrapids-connect](https://github.com/q-rapids/qrapids-connect) | Connectors | Apache Kafka connectors for ingesting data from heterogeneous data sources. It includes an example of an Apache Kafka Connector that collects Issues and Measures from Sonarqube. | Fraunhofer IESE |
| [qrapids-eval](https://github.com/q-rapids/qrapids-eval) | Quality model tool support | It defines a quality model by aggregating the raw data into metrics, and further on into factors. The quality model can be customised and/or replaced in different companies, working as a plug-in, allowing the creation of quality models based on expert knowledge or data mining analysis techniques. | Fraunhofer IESE |

**Note.** You can find how these components are interacting each other in the [Q-Rapids Tool Architecture](https://github.com/q-rapids/q-rapids/wiki/Q-Rapids-Tool-Architecture) wiki page.

# Knowledge

| Name | Description | URL | Lead Organization |
| ---| ---- | ----------- | -------------| 
| **QR Patterns Catalogue Configuration** | Guidelines for the configuration of the QR patterns catalogue used for generating QRs | | UPC|
| **QR Patterns Catalogue** | QR patterns catalogue used for generating QRs | | UPC|
| **SI Qualitative Estimation Configuration** | Guidelines for the creation of the estimation model for SI using Bayesian Netwoks (BN)| | UPC|
| **Quality Model Configuration** | Guidelines for the configuration of the Quality Model (SIs, Factors, Metrics)| | Fraunhofer |
| **Quality Model factors and metrics** | Files used to congifure factors and metrics defined in the QRapids Quality Model | | Fraunhofer & UoO |
| **Q-Rapids Process** | Development process to integrate the Q-Rapids tool for managing quality in ASD || UoO |

# Contributing
See the Q-Rapids Contribution Guidelines [here](https://github.com/q-rapids/q-rapids/blob/master/CONTRIBUTING.md).

# Glosary
| Name | Description |
| ---| ---- | 
| **ASD** | Agile Software Development |
| **QM** | Quality Model |
| **QR** | Quality Requirement |
| **SI** | Strategic Indicators|

*Note*: Some of the icons used in the Q-Rapids space are comming from https://pixabay.com/

