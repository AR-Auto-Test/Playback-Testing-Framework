package com.google.android.gms.vision;

import android.util.SparseArray;
import androidx.annotation.RecentlyNonNull;
import com.google.android.gms.vision.Detector;
import java.util.ArrayList;
import java.util.List;

/* compiled from: com.google.android.gms:play-services-vision-common@@19.1.3 */
/* loaded from: classes.dex */
public class MultiDetector extends Detector<Object> {
    private List<Detector<? extends Object>> zza;

    /* compiled from: com.google.android.gms:play-services-vision-common@@19.1.3 */
    /* loaded from: classes.dex */
    public static class Builder {
        private MultiDetector zza = new MultiDetector();

        @RecentlyNonNull
        public Builder add(@RecentlyNonNull Detector<? extends Object> detector) {
            this.zza.zza.add(detector);
            return this;
        }

        @RecentlyNonNull
        public MultiDetector build() {
            if (this.zza.zza.size() != 0) {
                return this.zza;
            }
            throw new RuntimeException("No underlying detectors added to MultiDetector.");
        }
    }

    private MultiDetector() {
        this.zza = new ArrayList();
    }

    @Override // com.google.android.gms.vision.Detector
    @RecentlyNonNull
    public SparseArray<Object> detect(@RecentlyNonNull Frame frame) {
        SparseArray<Object> sparseArray = new SparseArray<>();
        for (Detector<? extends Object> detector : this.zza) {
            SparseArray<? extends Object> detect = detector.detect(frame);
            for (int i = 0; i < detect.size(); i++) {
                int keyAt = detect.keyAt(i);
                if (sparseArray.get(keyAt) == null) {
                    sparseArray.append(keyAt, detect.valueAt(i));
                } else {
                    StringBuilder sb = new StringBuilder(104);
                    sb.append("Detection ID overlap for id = ");
                    sb.append(keyAt);
                    sb.append("  This means that one of the detectors is not using global IDs.");
                    throw new IllegalStateException(sb.toString());
                }
            }
        }
        return sparseArray;
    }

    @Override // com.google.android.gms.vision.Detector
    public boolean isOperational() {
        for (Detector<? extends Object> detector : this.zza) {
            if (!detector.isOperational()) {
                return false;
            }
        }
        return true;
    }

    @Override // com.google.android.gms.vision.Detector
    public void receiveFrame(@RecentlyNonNull Frame frame) {
        for (Detector<? extends Object> detector : this.zza) {
            detector.receiveFrame(frame);
        }
    }

    @Override // com.google.android.gms.vision.Detector
    public void release() {
        for (Detector<? extends Object> detector : this.zza) {
            detector.release();
        }
        this.zza.clear();
    }

    @Override // com.google.android.gms.vision.Detector
    public void setProcessor(@RecentlyNonNull Detector.Processor<Object> processor) {
        throw new UnsupportedOperationException("MultiDetector.setProcessor is not supported.  You should set a processor instance on each underlying detector instead.");
    }
}