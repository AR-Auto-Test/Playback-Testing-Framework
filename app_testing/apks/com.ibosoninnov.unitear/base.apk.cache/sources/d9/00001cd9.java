package com.google.ar.sceneform.rendering;

/* loaded from: classes.dex */
public interface LoadGltfListener {

    /* loaded from: classes.dex */
    public enum GltfLoadStage {
        LOAD_STAGE_NONE,
        FETCH_MATERIALS,
        DOWNLOAD_MODEL,
        CREATE_LOADER,
        ADD_MISSING_FILES,
        FINISHED_READING_FILES,
        CREATE_RENDERABLE
    }

    void onFinishedFetchingMaterials();

    void onFinishedLoadingModel(long j);

    void onFinishedReadingFiles(long j);

    void onReadingFilesFailed(Exception exc);

    void reportBytesDownloaded(long j);

    void setLoadingStage(GltfLoadStage gltfLoadStage);

    void setModelSize(float f2);
}