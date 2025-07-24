package com.google.android.gms.common.api.internal;

import android.app.Activity;
import java.lang.ref.WeakReference;
import java.util.ArrayList;
import java.util.List;

/* compiled from: com.google.android.gms:play-services-base@@17.4.0 */
/* loaded from: classes.dex */
public final class zaa extends ActivityLifecycleObserver {
    private final WeakReference<C0091zaa> zaa;

    public zaa(Activity activity) {
        this(C0091zaa.zab(activity));
    }

    @Override // com.google.android.gms.common.api.internal.ActivityLifecycleObserver
    public final ActivityLifecycleObserver onStopCallOnce(Runnable runnable) {
        C0091zaa c0091zaa = this.zaa.get();
        if (c0091zaa == null) {
            throw new IllegalStateException("The target activity has already been GC'd");
        }
        c0091zaa.zaa(runnable);
        return this;
    }

    /* compiled from: com.google.android.gms:play-services-base@@17.4.0 */
    /* renamed from: com.google.android.gms.common.api.internal.zaa$zaa  reason: collision with other inner class name */
    /* loaded from: classes.dex */
    public static class C0091zaa extends LifecycleCallback {
        private List<Runnable> zaa;

        private C0091zaa(LifecycleFragment lifecycleFragment) {
            super(lifecycleFragment);
            this.zaa = new ArrayList();
            this.mLifecycleFragment.addCallback("LifecycleObserverOnStop", this);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public final synchronized void zaa(Runnable runnable) {
            this.zaa.add(runnable);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public static C0091zaa zab(Activity activity) {
            C0091zaa c0091zaa;
            synchronized (activity) {
                LifecycleFragment fragment = LifecycleCallback.getFragment(activity);
                c0091zaa = (C0091zaa) fragment.getCallbackOrNull("LifecycleObserverOnStop", C0091zaa.class);
                if (c0091zaa == null) {
                    c0091zaa = new C0091zaa(fragment);
                }
            }
            return c0091zaa;
        }

        @Override // com.google.android.gms.common.api.internal.LifecycleCallback
        public void onStop() {
            List<Runnable> list;
            synchronized (this) {
                list = this.zaa;
                this.zaa = new ArrayList();
            }
            for (Runnable runnable : list) {
                runnable.run();
            }
        }
    }

    private zaa(C0091zaa c0091zaa) {
        this.zaa = new WeakReference<>(c0091zaa);
    }
}