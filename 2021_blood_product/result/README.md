# Directory for file output

## SHAP visualizations
Imagine a hypothetical patient undergoing a laparoscopic nephrectomy. Suppose this surgery has a historical transfusion rate of 1.3% at your hospital. Conventional MSBOS might not recommend a type and screen (T/S) due to the low overall rate of transfusion for this surgery. However, the model predicts a 7% risk of transfusion for this patient, due mostly to their low starting hematocrit and low platelet count. Thus, the model might recommend a T/S where the conventional method might not.

![image](https://github.com/sslou/publications/blob/main/2021_blood_product/result/shap_force_plot_0.jpeg)

See shap_explain.py for how this image is generated, or if you want to adjust patient variables and see how model predictions change.
