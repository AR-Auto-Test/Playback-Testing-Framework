package com.google.ar.sceneform.resources;

import com.google.ar.sceneform.resources.ResourceRegistry;
import com.google.ar.sceneform.utilities.Preconditions;
import java.lang.ref.WeakReference;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.function.BiFunction;

/* loaded from: classes.dex */
public class ResourceRegistry<T> implements ResourceHolder {
    private static final String TAG = "ResourceRegistry";
    private final Object lock = new Object();
    private final Map<Object, WeakReference<T>> registry = new HashMap();
    private final Map<Object, CompletableFuture<T>> futureRegistry = new HashMap();

    public /* synthetic */ Void a(Object obj, CompletableFuture completableFuture, Object obj2, Throwable th) {
        synchronized (this) {
            synchronized (this.lock) {
                if (this.futureRegistry.get(obj) == completableFuture) {
                    this.futureRegistry.remove(obj);
                    if (th == null) {
                        this.registry.put(obj, new WeakReference<>(obj2));
                    }
                }
            }
        }
        return null;
    }

    @Override // com.google.ar.sceneform.resources.ResourceHolder
    public void destroyAllResources() {
        synchronized (this.lock) {
            Iterator<Map.Entry<Object, CompletableFuture<T>>> it = this.futureRegistry.entrySet().iterator();
            while (it.hasNext()) {
                it.remove();
                CompletableFuture<T> value = it.next().getValue();
                if (!value.isDone()) {
                    value.cancel(true);
                }
            }
            this.registry.clear();
        }
    }

    public CompletableFuture<T> get(Object obj) {
        Preconditions.checkNotNull(obj, "Parameter 'id' was null.");
        synchronized (this.lock) {
            WeakReference<T> weakReference = this.registry.get(obj);
            if (weakReference != null) {
                T t = weakReference.get();
                if (t != null) {
                    return CompletableFuture.completedFuture(t);
                }
                this.registry.remove(obj);
            }
            return this.futureRegistry.get(obj);
        }
    }

    @Override // com.google.ar.sceneform.resources.ResourceHolder
    public long reclaimReleasedResources() {
        return 0L;
    }

    public void register(final Object obj, final CompletableFuture<T> completableFuture) {
        Preconditions.checkNotNull(obj, "Parameter 'id' was null.");
        Preconditions.checkNotNull(completableFuture, "Parameter 'futureResource' was null.");
        if (completableFuture.isDone()) {
            if (completableFuture.isCompletedExceptionally()) {
                return;
            }
            Object checkNotNull = Preconditions.checkNotNull(completableFuture.getNow(null));
            synchronized (this.lock) {
                this.registry.put(obj, new WeakReference<>(checkNotNull));
                this.futureRegistry.remove(obj);
            }
            return;
        }
        synchronized (this.lock) {
            this.futureRegistry.put(obj, completableFuture);
            this.registry.remove(obj);
        }
        completableFuture.handle((BiFunction) new BiFunction() { // from class: c.d.b.a.r.a
            @Override // java.util.function.BiFunction
            public final Object apply(Object obj2, Object obj3) {
                ResourceRegistry.this.a(obj, completableFuture, obj2, (Throwable) obj3);
                return null;
            }
        });
    }
}