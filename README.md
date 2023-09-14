# CET
# Effect of distance measures on confidences of t-SNE embeddings and its implications on clustering for scRNA-seq data
## Busra Ozgode Yigin and Gorkem Saygili
### Cognitive Sciences and Artificial Intelligence, Tilburg School of Humanities and Digital Sciences, Tilburg University, The Netherlands.

Abstract:
Arguably one of the most famous dimensionality reduction algorithms of today is t-distributed stochastic neighbor embedding (t-SNE). Although being widely used for the visualization of scRNA-seq data, it is prone to errors as any algorithm and may lead to inaccurate interpretations of the visualized data. A reasonable way to avoid misinterpretations is to quantify the reliability of the visualizations. The focus of this work is first to find the best possible way to predict sample-based confidence scores for t-SNE embeddings and next, to use these confidence scores to improve the clustering algorithms. We adopt an RF regression algorithm using seven distance measures as features for having the sample-based confidence scores with a variety of different distance measures. The best configuration is used to assess the clustering improvement using K-means and Density-Based Spatial Clustering of Applications with Noise (DBSCAN) based on Adjusted Rank Index (ARI), Normalized Mutual Information (NMI), and accuracy (ACC) scores. The experimental results show that distance measures have a considerable effect on the precision of confidence scores and clustering performance can be improved substantially if these confidence scores are incorporated before the clustering algorithm. Our findings reveal the usefulness of these confidence scores on downstream analyses for scRNA-seq data.

This code is the code of our journal publication:

***[1] B. Ozgode Yigin and G. Saygili, "Effect of distance measures on confidences of t-SNE embeddings and itsimplications on clustering for scRNA-seq data", Nature Scientific Reports, 2023.***


***Please cite our paper [[1]](https://www.nature.com/articles/s41598-023-32966-x) in case you use the code.***

Created by Busra Ozgode Yigin and Gorkem Saygili on 14-04-23.

Datasets:
1) Allen Mouse Brain (AMB18)
2) Baron Mouse
3) Baron Human
4) Cellbench
5) Segerstolpe
6) Muraro

***Important Note: This code is under MIT License:***

***THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.***

How to use:
- You can run run_experiments function to run our experiments mentioned on the paper.
