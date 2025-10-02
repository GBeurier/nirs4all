# '''
# External prediction storage functions for the simplified model controller.
# Predictions are stored in external prediction store passed as argument.
# '''

# def store_training_predictions(prediction_store, dataset_name, pipeline_name,
#                                model_name, model_id, model_uuid,
#                                fold, step, op_counter,
#                                predictions_train, predictions_valid,
#                                train_indices, valid_indices,
#                                custom_model_name=None, context=None, dataset=None,
#                                pipeline_path="", config_path="", config_id="",
#                                model_file_name=None):
#     '''Store training predictions in external prediction store. Data is expected to be in unscaled format.'''

#     # Enhanced metadata with model paths and identifiers
#     enhanced_metadata = {
#         'step': step,
#         'op_counter': op_counter,
#         'model_id': model_id,
#         'config_id': config_id,
#         'pipeline_path': pipeline_path,
#         'config_path': config_path,
#         'enhanced_model_name': f"{model_name}_{model_id}" if model_id else model_name,
#         'model_path': f"{config_path}/{model_uuid}.pkl" if config_path else f"{model_uuid}.pkl"
#     }

#     # Store validation predictions (the ones used for averaging)
#     if predictions_valid is not None:
#         y_true_val = predictions_valid.get('y_true') if isinstance(predictions_valid, dict) else None
#         y_pred_val = predictions_valid.get('y_pred') if isinstance(predictions_valid, dict) else predictions_valid

#         prediction_store.add_prediction(
#             dataset=dataset_name,
#             pipeline=pipeline_name,
#             model=model_name,  # Base model class name
#             partition='val',
#             y_true=y_true_val,
#             y_pred=y_pred_val,
#             sample_indices=valid_indices,  # Add sample_indices
#             fold_idx=fold,
#             real_model=model_uuid,  # Full model identifier
#             custom_model_name=custom_model_name,
#             pipeline_path=pipeline_path,
#             metadata=enhanced_metadata
#         )

#     # Store training predictions if available
#     if predictions_train is not None:
#         y_true_train = predictions_train.get('y_true') if isinstance(predictions_train, dict) else None
#         y_pred_train = predictions_train.get('y_pred') if isinstance(predictions_train, dict) else predictions_train

#         prediction_store.add_prediction(
#             dataset=dataset_name,
#             pipeline=pipeline_name,
#             model=model_name,
#             partition='train',
#             y_true=y_true_train,
#             y_pred=y_pred_train,
#             sample_indices=train_indices,  # Add sample_indices
#             fold_idx=fold,
#             real_model=model_uuid,
#             custom_model_name=custom_model_name,
#             pipeline_path=pipeline_path,
#             metadata=enhanced_metadata
#         )

# def store_test_predictions(prediction_store, dataset_name, pipeline_name,
#                            model_name, model_id, model_uuid,
#                            fold, step, op_counter,
#                            y_true, y_pred, test_indices,
#                            custom_model_name=None, context=None, dataset=None,
#                            pipeline_path="", config_path="", config_id="",
#                            model_file_name=None):
#     '''Store test predictions in external prediction store. Data is expected to be in unscaled format.'''

#     # Enhanced metadata with model paths and identifiers
#     enhanced_metadata = {
#         'step': step,
#         'op_counter': op_counter,
#         'model_id': model_id,
#         'config_id': config_id,
#         'pipeline_path': pipeline_path,
#         'config_path': config_path,
#         'enhanced_model_name': f"{model_name}_{model_id}" if model_id else model_name,
#         'model_path': f"{config_path}/{model_file_name}" if (model_file_name and config_path) else f"{config_path}/{model_uuid}.pkl" if config_path else f"{model_uuid}.pkl"
#     }

#     prediction_store.add_prediction(
#         dataset=dataset_name,
#         pipeline=pipeline_name,
#         model=model_name,
#         partition='test',
#         y_true=y_true,
#         y_pred=y_pred,
#         sample_indices=test_indices,  # Add sample_indices
#         fold_idx=fold,
#         real_model=model_uuid,
#         custom_model_name=custom_model_name,
#         pipeline_path=pipeline_path,
#         metadata=enhanced_metadata
#     )

# def store_virtual_model_predictions(prediction_store, dataset_name, pipeline_name,
#                                     model_name, model_id, model_uuid,
#                                     partition, fold_idx, step,
#                                     y_true, y_pred, test_indices,
#                                     custom_model_name=None, context=None, dataset=None,
#                                     pipeline_path="", config_path="", config_id="",
#                                     virtual_metadata=None):
#     '''Store virtual model predictions (like avg, w-avg) in external prediction store. Data is expected to be in unscaled format.'''

#     # Enhanced metadata for virtual models
#     enhanced_metadata = {
#         'is_virtual_model': True,
#         'virtual_type': fold_idx,
#         'step': step,
#         'model_id': model_id,
#         'config_id': config_id,
#         'pipeline_path': pipeline_path,
#         'config_path': config_path,
#         'enhanced_model_name': f"{model_name}_{model_id}_{fold_idx}" if model_id else f"{model_name}_{fold_idx}",
#         'model_path': f"{config_path}/{model_uuid}.pkl" if config_path else f"{model_uuid}.pkl"
#     }

#     # Add virtual model specific metadata (weights, constituent models, etc.)
#     if virtual_metadata:
#         enhanced_metadata.update(virtual_metadata)

#     prediction_store.add_prediction(
#         dataset=dataset_name,
#         pipeline=pipeline_name,
#         model=model_name,
#         partition=partition,
#         y_true=y_true,
#         y_pred=y_pred,
#         sample_indices=test_indices,  # Add sample_indices
#         fold_idx=fold_idx,  # 'avg', 'w-avg', or integer
#         real_model=model_uuid,
#         custom_model_name=custom_model_name,
#         pipeline_path=pipeline_path,
#         metadata=enhanced_metadata
#     )

# def get_top_k_models(prediction_store, dataset_name, k=10, metric='mse', partition='test'):
#     '''Get top K models by metric.'''
#     all_predictions = prediction_store.get_predictions(
#         dataset=dataset_name,
#         partition=partition
#     )

#     # Calculate scores and sort
#     scored_predictions = []
#     for key, pred_data in all_predictions.items():
#         # Calculate metric if not stored
#         scores = prediction_store.calculate_scores_for_predictions({key: pred_data})
#         if key in scores and metric in scores[key]:
#             scored_predictions.append((key, pred_data, scores[key][metric]))

#     # Sort by metric (ascending for error metrics, descending for accuracy metrics)
#     ascending = metric in ['mse', 'rmse', 'mae']
#     scored_predictions.sort(key=lambda x: x[2], reverse=not ascending)

#     return scored_predictions[:k]
