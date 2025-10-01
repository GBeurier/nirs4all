'''
External prediction storage functions for the simplified model controller.
Predictions are stored in external prediction store passed as argument.
'''

def store_training_predictions(prediction_store, dataset_name, pipeline_name,
                             model_name, model_id, model_uuid,
                             fold, step, op_counter,
                             predictions_train, predictions_valid,
                             train_indices, valid_indices,
                             custom_model_name=None):
    '''Store training predictions in external prediction store.'''

    # Store validation predictions (the ones used for averaging)
    if predictions_valid is not None:
        prediction_store.add_prediction(
            dataset=dataset_name,
            pipeline=pipeline_name,
            model=model_name,  # Base model class name
            partition='val',
            y_true=predictions_valid.get('y_true') if isinstance(predictions_valid, dict) else None,
            y_pred=predictions_valid.get('y_pred') if isinstance(predictions_valid, dict) else predictions_valid,
            sample_indices=valid_indices,  # Add sample_indices
            fold_idx=fold,
            real_model=model_uuid,  # Full model identifier
            custom_model_name=custom_model_name
        )

    # Store training predictions if available
    if predictions_train is not None:
        prediction_store.add_prediction(
            dataset=dataset_name,
            pipeline=pipeline_name,
            model=model_name,
            partition='train',
            y_true=predictions_train.get('y_true') if isinstance(predictions_train, dict) else None,
            y_pred=predictions_train.get('y_pred') if isinstance(predictions_train, dict) else predictions_train,
            sample_indices=train_indices,  # Add sample_indices
            fold_idx=fold,
            real_model=model_uuid,
            custom_model_name=custom_model_name
        )

def store_test_predictions(prediction_store, dataset_name, pipeline_name,
                         model_name, model_id, model_uuid,
                         fold, step, op_counter,
                         y_true, y_pred, test_indices,
                         custom_model_name=None):
    '''Store test predictions in external prediction store.'''
    prediction_store.add_prediction(
        dataset=dataset_name,
        pipeline=pipeline_name,
        model=model_name,
        partition='test',
        y_true=y_true,
        y_pred=y_pred,
        sample_indices=test_indices,  # Add sample_indices
        fold_idx=fold,
        real_model=model_uuid,
        custom_model_name=custom_model_name
    )

def store_virtual_model_predictions(prediction_store, dataset_name, pipeline_name,
                                  model_name, model_id, model_uuid,
                                  partition, fold_idx, step,
                                  y_true, y_pred, test_indices,
                                  custom_model_name=None):
    '''Store virtual model predictions (like avg, w-avg) in external prediction store.'''
    prediction_store.add_prediction(
        dataset=dataset_name,
        pipeline=pipeline_name,
        model=model_name,
        partition=partition,
        y_true=y_true,
        y_pred=y_pred,
        sample_indices=test_indices,  # Add sample_indices
        fold_idx=fold_idx,  # 'avg', 'w-avg', or integer
        real_model=model_uuid,
        custom_model_name=custom_model_name,
        metadata={'is_virtual_model': True, 'virtual_type': fold_idx}
    )

def get_top_k_models(prediction_store, dataset_name, k=10, metric='mse', partition='test'):
    '''Get top K models by metric.'''
    all_predictions = prediction_store.get_predictions(
        dataset=dataset_name,
        partition=partition
    )

    # Calculate scores and sort
    scored_predictions = []
    for key, pred_data in all_predictions.items():
        # Calculate metric if not stored
        scores = prediction_store.calculate_scores_for_predictions({key: pred_data})
        if key in scores and metric in scores[key]:
            scored_predictions.append((key, pred_data, scores[key][metric]))

    # Sort by metric (ascending for error metrics, descending for accuracy metrics)
    ascending = metric in ['mse', 'rmse', 'mae']
    scored_predictions.sort(key=lambda x: x[2], reverse=not ascending)

    return scored_predictions[:k]
