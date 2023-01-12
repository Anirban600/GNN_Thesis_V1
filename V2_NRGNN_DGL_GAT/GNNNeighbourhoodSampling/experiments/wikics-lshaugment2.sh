## Naming Convention --> lshaugment_neighbours1_neighbours2_lshneighbourssample_nlayer_numlshneighbours_atleast_searchradius_includeNeighbourhood

python -m graphsage.main --dataset wikics --epochs 150 --rand_walk False --neighbours1 20 --neighbours2 20 --aggregator mean --attention normal --iter 2 --outdir /home/thummala/graphsage-pytorch/res/wikics/lshaugment_20_20_10_2_20_true_2_True --n_layers 2 --includenodefeats no --typewalk default --augment_khop True --n_lsh_neighbours_sample 10 --search_radius 2 --atleast True --num_lsh_neighbours 20 --save_predictions True --includeNeighbourhood True

python -m graphsage.main --dataset wikics --epochs 150 --rand_walk False --neighbours1 20 --neighbours2 20 --aggregator mean --attention normal --iter 2 --outdir /home/thummala/graphsage-pytorch/res/wikics/lshaugment_20_20_10_2_20_false_2_True --n_layers 2 --includenodefeats no --typewalk default --augment_khop True --n_lsh_neighbours_sample 10 --search_radius 2 --atleast False --num_lsh_neighbours 20 --save_predictions True --includeNeighbourhood True

python -m graphsage.main --dataset wikics --epochs 150 --rand_walk False --neighbours1 20 --neighbours2 20 --aggregator mean --attention normal --iter 2 --outdir /home/thummala/graphsage-pytorch/res/wikics/lshaugment_20_20_20_2_20_true_2_True --n_layers 2 --includenodefeats no --typewalk default --augment_khop True --n_lsh_neighbours_sample 20 --search_radius 2 --atleast True --num_lsh_neighbours 20 --save_predictions True --includeNeighbourhood True

python -m graphsage.main --dataset wikics --epochs 150 --rand_walk False --neighbours1 20 --neighbours2 20 --aggregator mean --attention normal --iter 2 --outdir /home/thummala/graphsage-pytorch/res/wikics/lshaugment_20_20_30_2_30_true_2_True --n_layers 2 --includenodefeats no --typewalk default --augment_khop True --n_lsh_neighbours_sample 30 --search_radius 2 --atleast True --num_lsh_neighbours 30 --save_predictions True --includeNeighbourhood True

python -m graphsage.main --dataset wikics --epochs 150 --rand_walk False --neighbours1 10 --neighbours2 10 --aggregator mean --attention normal --iter 2 --outdir /home/thummala/graphsage-pytorch/res/wikics/lshaugment_10_10_20_2_20_true_2_True --n_layers 2 --includenodefeats no --typewalk default --augment_khop True --n_lsh_neighbours_sample 20 --search_radius 2 --atleast True --num_lsh_neighbours 20 --save_predictions True --includeNeighbourhood True

python -m graphsage.main --dataset wikics --epochs 150 --rand_walk False --neighbours1 20 --neighbours2 20 --aggregator mean --attention normal --iter 2 --outdir /home/thummala/graphsage-pytorch/res/wikics/lshaugment_20_20_5_2_10_true_2_True --n_layers 2 --includenodefeats no --typewalk default --augment_khop True --n_lsh_neighbours_sample 5 --search_radius 2 --atleast True --num_lsh_neighbours 10 --save_predictions True --includeNeighbourhood True

python -m graphsage.main --dataset wikics --epochs 150 --rand_walk False --neighbours1 20 --neighbours2 20 --aggregator mean --attention normal --iter 2 --outdir /home/thummala/graphsage-pytorch/res/wikics/lshaugment_20_20_na_2_20_false_2_True --n_layers 2 --includenodefeats no --typewalk default --augment_khop True --search_radius 2 --atleast False --num_lsh_neighbours 20 --save_predictions True --includeNeighbourhood True

python -m graphsage.main --dataset wikics --epochs 150 --rand_walk False --neighbours1 20 --neighbours2 20 --aggregator mean --attention normal --iter 2 --outdir /home/thummala/graphsage-pytorch/res/wikics/lshaugment_20_20_na_2_20_false_3_True --n_layers 2 --includenodefeats no --typewalk default --augment_khop True --search_radius 3 --atleast False --num_lsh_neighbours 20 --save_predictions True --includeNeighbourhood True

python -m graphsage.main --dataset wikics --epochs 150 --rand_walk False --neighbours1 20 --neighbours2 20 --aggregator mean --attention normal --iter 2 --outdir /home/thummala/graphsage-pytorch/res/wikics/lshaugment_20_20_20_2_20_false_3_True --n_layers 2 --includenodefeats no --typewalk default --augment_khop True --n_lsh_neighbours_sample 20 --search_radius 3 --atleast False --num_lsh_neighbours 20 --save_predictions True --includeNeighbourhood True
