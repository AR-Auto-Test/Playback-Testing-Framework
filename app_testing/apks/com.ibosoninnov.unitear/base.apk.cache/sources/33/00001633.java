package com.google.android.gms.vision;

import android.util.SparseArray;
import androidx.annotation.RecentlyNonNull;
import c.b.a.a.a;
import com.google.android.gms.vision.Detector;
import java.util.HashSet;
import java.util.Iterator;

/* compiled from: com.google.android.gms:play-services-vision-common@@19.1.3 */
/* loaded from: classes.dex */
public class MultiProcessor<T> implements Detector.Processor<T> {
    private Factory<T> zza;
    private SparseArray<zza> zzb;
    private int zzc;

    /* compiled from: com.google.android.gms:play-services-vision-common@@19.1.3 */
    /* loaded from: classes.dex */
    public static class Builder<T> {
        private MultiProcessor<T> zza;

        public Builder(@RecentlyNonNull Factory<T> factory) {
            MultiProcessor<T> multiProcessor = new MultiProcessor<>();
            this.zza = multiProcessor;
            if (factory != null) {
                ((MultiProcessor) multiProcessor).zza = factory;
                return;
            }
            throw new IllegalArgumentException("No factory supplied.");
        }

        @RecentlyNonNull
        public MultiProcessor<T> build() {
            return this.zza;
        }

        @RecentlyNonNull
        public Builder<T> setMaxGapFrames(int i) {
            if (i >= 0) {
                ((MultiProcessor) this.zza).zzc = i;
                return this;
            }
            throw new IllegalArgumentException(a.g(28, "Invalid max gap: ", i));
        }
    }

    /* compiled from: com.google.android.gms:play-services-vision-common@@19.1.3 */
    /* loaded from: classes.dex */
    public interface Factory<T> {
        @RecentlyNonNull
        Tracker<T> create(@RecentlyNonNull T t);
    }

    /* compiled from: com.google.android.gms:play-services-vision-common@@19.1.3 */
    /* loaded from: classes.dex */
    public class zza {
        private Tracker<T> zza;
        private int zzb;

        private zza(MultiProcessor multiProcessor) {
            this.zzb = 0;
        }

        public static /* synthetic */ int zzb(zza zzaVar) {
            int i = zzaVar.zzb;
            zzaVar.zzb = i + 1;
            return i;
        }
    }

    private MultiProcessor() {
        this.zzb = new SparseArray<>();
        this.zzc = 3;
    }

    @Override // com.google.android.gms.vision.Detector.Processor
    public void receiveDetections(@RecentlyNonNull Detector.Detections<T> detections) {
        SparseArray<T> detectedItems = detections.getDetectedItems();
        for (int i = 0; i < detectedItems.size(); i++) {
            int keyAt = detectedItems.keyAt(i);
            T valueAt = detectedItems.valueAt(i);
            if (this.zzb.get(keyAt) == null) {
                zza zzaVar = new zza();
                zzaVar.zza = this.zza.create(valueAt);
                zzaVar.zza.onNewItem(keyAt, valueAt);
                this.zzb.append(keyAt, zzaVar);
            }
        }
        SparseArray<T> detectedItems2 = detections.getDetectedItems();
        HashSet hashSet = new HashSet();
        for (int i2 = 0; i2 < this.zzb.size(); i2++) {
            int keyAt2 = this.zzb.keyAt(i2);
            if (detectedItems2.get(keyAt2) == null) {
                zza valueAt2 = this.zzb.valueAt(i2);
                zza.zzb(valueAt2);
                if (valueAt2.zzb >= this.zzc) {
                    valueAt2.zza.onDone();
                    hashSet.add(Integer.valueOf(keyAt2));
                } else {
                    valueAt2.zza.onMissing(detections);
                }
            }
        }
        Iterator it = hashSet.iterator();
        while (it.hasNext()) {
            this.zzb.delete(((Integer) it.next()).intValue());
        }
        SparseArray<T> detectedItems3 = detections.getDetectedItems();
        for (int i3 = 0; i3 < detectedItems3.size(); i3++) {
            int keyAt3 = detectedItems3.keyAt(i3);
            T valueAt3 = detectedItems3.valueAt(i3);
            zza zzaVar2 = this.zzb.get(keyAt3);
            zzaVar2.zzb = 0;
            zzaVar2.zza.onUpdate(detections, valueAt3);
        }
    }

    @Override // com.google.android.gms.vision.Detector.Processor
    public void release() {
        for (int i = 0; i < this.zzb.size(); i++) {
            this.zzb.valueAt(i).zza.onDone();
        }
        this.zzb.clear();
    }
}