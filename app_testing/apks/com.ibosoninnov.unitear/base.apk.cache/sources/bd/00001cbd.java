package com.google.ar.sceneform.rendering;

import com.google.ar.sceneform.resources.ResourceHolder;
import java.lang.ref.ReferenceQueue;
import java.util.HashSet;
import java.util.Iterator;

/* loaded from: classes.dex */
public class CleanupRegistry<T> implements ResourceHolder {
    private final HashSet<CleanupItem<T>> cleanupItemHashSet;
    private final ReferenceQueue<T> referenceQueue;

    public CleanupRegistry() {
        this(new HashSet(), new ReferenceQueue());
    }

    @Override // com.google.ar.sceneform.resources.ResourceHolder
    public void destroyAllResources() {
        Iterator<CleanupItem<T>> it = this.cleanupItemHashSet.iterator();
        while (it.hasNext()) {
            it.remove();
            it.next().run();
        }
    }

    @Override // com.google.ar.sceneform.resources.ResourceHolder
    public long reclaimReleasedResources() {
        CleanupItem cleanupItem = (CleanupItem) this.referenceQueue.poll();
        while (cleanupItem != null) {
            if (this.cleanupItemHashSet.contains(cleanupItem)) {
                cleanupItem.run();
                this.cleanupItemHashSet.remove(cleanupItem);
            }
            cleanupItem = (CleanupItem) this.referenceQueue.poll();
        }
        return this.cleanupItemHashSet.size();
    }

    public void register(T t, Runnable runnable) {
        this.cleanupItemHashSet.add(new CleanupItem<>(t, this.referenceQueue, runnable));
    }

    public CleanupRegistry(HashSet<CleanupItem<T>> hashSet, ReferenceQueue<T> referenceQueue) {
        this.cleanupItemHashSet = hashSet;
        this.referenceQueue = referenceQueue;
    }
}