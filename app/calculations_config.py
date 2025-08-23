CALCULATIONS_CONFIG = {
  "workshop": [
    {
      "name": "ERP Envisioning Workshop",
      "formula": "min((7.5% of ACV), no_of_hour * market_rate, 6000)",
      "blocker": [
        "country"
      ],
      "form_fields": [
        {
          "field_name": "acv",
          "about": "The estimated value of annual billed revenue from the customer",
          "label": "Annual Contract Value",
          "value": 0
        },
        {
          "field_name": "no_of_hour",
          "about": "Estimated number of hours you will need to complete this workshop",
          "label": "Approx number of hours",
          "value": 0
        }
      ]
    },
    {
      "name": "CRM Envisioning Workshop",
      "formula": "min((7.5% of ACV), no_of_hour * market_rate, 6000)",
      "blocker": [
        "country"
      ],
      "form_fields": [
        {
          "field_name": "acv",
          "about": "The estimated value of annual billed revenue from the customer",
          "label": "Annual Contract Value",
          "value": 0
        },
        {
          "field_name": "no_of_hour",
          "about": "Estimated number of hours you will need to complete this workshop",
          "label": "Approx number of hours",
          "value": 0
        }
      ]
    }
  ],
  "csp_transaction": [
    {
      "name": "D365 CSP Core",
      "formula": "core_billed_revenue * 0.04",
      "blocker": [],
      "form_fields": [
        {
          "field_name": "core_billed_revenue",
          "about": "The estimated annual revenue billed from the customer",
          "label": "Annual billed revenue from customer",
          "value": 0
        }
      ]
    },
    {
      "name": "D365 CSP Global Strategic Product Accelerator – Tier 1 (Finance & Supply Chain)",
      "formula": "tier_1_billed_revenue * 0.07",
      "blocker": [],
      "form_fields": [
        {
          "field_name": "tier_1_billed_revenue",
          "about": "Billed revenue that can be attributed to strategic products (Finance & Supply Chain)",
          "label": "Annual billed revenue from Global Strategic Product Accelerator Tier 1",
          "value": 0
        }
      ]
    },
    {
      "name": "D365 CSP Global Strategic Product Accelerator – Tier 2 (Business Central)",
      "formula": "tier_2_billed_revenue * 0.08",
      "blocker": [],
      "form_fields": [
        {
          "field_name": "tier_2_billed_revenue",
          "about": "Billed revenue that can be attributed to strategic products (Business Central)",
          "label": "Annual billed revenue from Global Strategic Product Accelerator Tier 2",
          "value": 0
        }
      ]
    },
    {
      "name": "D365 CSP Growth Accelerator",
      "formula": "(current_year_billed_revenue - last_year_billed_revenue) * 0.08",
      "blocker": [],
      "form_fields": [
        {
          "field_name": "current_year_billed_revenue",
          "about": "Revenue billed from the customer in the current year",
          "label": "Billed revenue from customer in current year",
          "value": 0
        },
        {
          "field_name": "last_year_billed_revenue",
          "about": "Revenue billed from the customer same month last year",
          "label": "Billed revenue from customer in previous year",
          "value": 0
        }
      ]
    }
  ],
  "spd_eligibility": {
    "smb": [
      {
        "name": "Performance Category",
        "formula": "min((current_year_workloads - last_year_workloads ) * 3 , 15)",
        "blocker": [],
        "form_fields": [
          {
            "field_name": "current_year_workloads",
            "about": "The total number of eligible workloads in last month of current year",
            "label": "Current Year Eligible workloads",
            "value": 0
          },
          {
            "field_name": "last_year_workloads",
            "about": "The total number of eligible workloads in same month last year",
            "label": "Last Year Eligible Workloads",
            "value": 0
          }
        ]
      },
      {
        "name": "Skilling Category",
        "formula": "min(no_of_intermediate_certifications,20)) + (min((no_of_advanced_certifications *7.5),15)",
        "blocker": [],
        "form_fields": [
          {
            "field_name": "no_of_intermediate_certifications",
            "about": "Number of individuals who have completed intermediate certifications",
            "label": "Number of intermediate certifications completed",
            "value": 0
          },
          {
            "field_name": "no_of_advanced_certifications",
            "about": "Number of individuals who have completed advanced certifications",
            "label": "Number of advanced certifications completed",
            "value": 0
          }
        ]
      },
      {
        "name": "Customer Sucess Category",
        "formula": "usage_metric  + deployment_metric",
        "blocker": [],
        "sub_module": [
          {
            "name": "Usage Metric Score",
            "formula": "usage_metric = min((current_year_usage - last_year_usage)/(last_year_usage*100),30)",
            "form_fields": [
              {
                "field_name": "current_year_usage",
                "about": "Eligible usage in last month of current year",
                "label": "Current Year Usage",
                "value": 0
              },
              {
                "field_name": "last_year_usage",
                "about": "Eligible usage in same month last year",
                "label": "Last Year Usage",
                "value": 0
              }
            ]
          },
          {
            "name": "Deployment Metric Score",
            "formula": "deployment_metric = min((current_year_workload - last_year_workload)*4,20)",
            "form_fields": [
              {
                "field_name": "current_year_workload",
                "about": "Number of eligible deployments made in last month of current year",
                "label": "Current Year Deployments",
                "value": 0
              },
              {
                "field_name": "last_year_workload",
                "about": "Number of eligible deployments made in same month last year",
                "label": "Last Year Deployments",
                "value": 0
              }
            ]
          }
        ]
      }
    ],
    "enterprise": [
      {
        "name": "Performance Category",
        "formula": "min((current_year_workloads - last_year_workloads ) * 3 , 15)",
        "blocker": [],
        "form_fields": [
          {
            "field_name": "current_year_workloads",
            "about": "Eligible workloads in last month of current year",
            "label": "Current Year Eligible Workloads",
            "value": 0
          },
          {
            "field_name": "last_year_workloads",
            "about": "Eligible workloads in same month of last year",
            "label": "Last Year Eligible Workloads",
            "value": 0
          }
        ]
      },
      {
        "name": "Skilling Category",
        "formula": "min(no_of_intermediate_certifications,20)) + (min((no_of_advanced_certifications *2.4),15)",
        "blocker": [],
        "form_fields": [
          {
            "field_name": "no_of_intermediate_certifications",
            "about": "Number of individuals who have completed intermediate certifications",
            "label": "Number of intermediate certifications completed",
            "value": 0
          },
          {
            "field_name": "no_of_advanced_certifications",
            "about": "Number of individuals who have completed advanced certifications",
            "label": "Number of advanced certifications completed",
            "value": 0
          }
        ]
      },
      {
        "name": "Customer Sucess Category",
        "formula": "usage_metric  + deployment_metric",
        "blocker": [],
        "sub_module": [
          {
            "name": "Usage Metric Score",
            "formula": "usage_metric = min((current_year_usage - last_year_usage)/(last_year_usage*100),30)",
            "form_fields": [
              {
                "field_name": "current_year_usage",
                "about": "Eligible usage in last month of current year",
                "label": "Current Year Usage",
                "value": 0
              },
              {
                "field_name": "last_year_usage",
                "about": "Eligible usage in same month of last year",
                "label": "Last Year Usage",
                "value": 0
              }
            ]
          },
          {
            "name": "Deployment Metric Score",
            "formula": "deployment_metric = min((current_year_workload - last_year_workload)*4,20)",
            "form_fields": [
              {
                "field_name": "current_year_workload",
                "about": "Number of eligible workloads deployed in last month of current year",
                "label": "Current Year Deployments",
                "value": 0
              },
              {
                "field_name": "last_year_workload",
                "about": "Number of eligible workloads deployed in same month of last year",
                "label": "Last Year Deployments",
                "value": 0
              }
            ]
          }
        ]
      }
    ]
  }
}