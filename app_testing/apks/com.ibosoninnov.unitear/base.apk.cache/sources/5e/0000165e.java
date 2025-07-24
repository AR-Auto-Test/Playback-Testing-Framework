package com.google.android.gms.vision.face;

import android.content.Context;
import android.graphics.Bitmap;
import android.media.Image;
import android.util.Log;
import android.util.SparseArray;
import androidx.annotation.RecentlyNonNull;
import c.b.a.a.a;
import com.google.android.gms.common.internal.Preconditions;
import com.google.android.gms.internal.vision.zzs;
import com.google.android.gms.internal.vision.zzw;
import com.google.android.gms.vision.Detector;
import com.google.android.gms.vision.Frame;
import com.google.android.gms.vision.face.internal.client.zzb;
import com.google.android.gms.vision.face.internal.client.zzf;
import com.google.android.gms.vision.zzc;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.nio.ByteBuffer;
import java.util.HashSet;

/* compiled from: com.google.android.gms:play-services-vision@@20.1.3 */
/* loaded from: classes.dex */
public final class FaceDetector extends Detector<Face> {
    public static final int ACCURATE_MODE = 1;
    public static final int ALL_CLASSIFICATIONS = 1;
    public static final int ALL_LANDMARKS = 1;
    public static final int CONTOUR_LANDMARKS = 2;
    public static final int FAST_MODE = 0;
    public static final int NO_CLASSIFICATIONS = 0;
    public static final int NO_LANDMARKS = 0;
    public static final int SELFIE_MODE = 2;
    private final zzc zza;
    private final zzb zzb;
    private final Object zzc;
    private boolean zzd;

    /* compiled from: com.google.android.gms:play-services-vision@@20.1.3 */
    /* loaded from: classes.dex */
    public static class Builder {
        private final Context zza;
        private int zzb = 0;
        private boolean zzc = false;
        private int zzd = 0;
        private boolean zze = true;
        private int zzf = 0;
        private float zzg = -1.0f;

        public Builder(@RecentlyNonNull Context context) {
            this.zza = context;
        }

        @RecentlyNonNull
        public FaceDetector build() {
            zzf zzfVar = new zzf();
            zzfVar.zza = this.zzf;
            zzfVar.zzb = this.zzb;
            zzfVar.zzc = this.zzd;
            zzfVar.zzd = this.zzc;
            zzfVar.zze = this.zze;
            zzfVar.zzf = this.zzg;
            if (FaceDetector.zzb(zzfVar)) {
                return new FaceDetector(new zzb(this.zza, zzfVar));
            }
            throw new IllegalArgumentException("Invalid build options");
        }

        @RecentlyNonNull
        public Builder setClassificationType(int i) {
            if (i != 0 && i != 1) {
                throw new IllegalArgumentException(a.g(40, "Invalid classification type: ", i));
            }
            this.zzd = i;
            return this;
        }

        @RecentlyNonNull
        public Builder setLandmarkType(int i) {
            if (i != 0 && i != 1 && i != 2) {
                throw new IllegalArgumentException(a.g(34, "Invalid landmark type: ", i));
            }
            this.zzb = i;
            return this;
        }

        @RecentlyNonNull
        public Builder setMinFaceSize(float f2) {
            if (f2 >= StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD && f2 <= 1.0f) {
                this.zzg = f2;
                return this;
            }
            StringBuilder sb = new StringBuilder(47);
            sb.append("Invalid proportional face size: ");
            sb.append(f2);
            throw new IllegalArgumentException(sb.toString());
        }

        @RecentlyNonNull
        public Builder setMode(int i) {
            if (i != 0 && i != 1 && i != 2) {
                throw new IllegalArgumentException(a.g(25, "Invalid mode: ", i));
            }
            this.zzf = i;
            return this;
        }

        @RecentlyNonNull
        public Builder setProminentFaceOnly(boolean z) {
            this.zzc = z;
            return this;
        }

        @RecentlyNonNull
        public Builder setTrackingEnabled(boolean z) {
            this.zze = z;
            return this;
        }
    }

    private FaceDetector() {
        this.zza = new zzc();
        this.zzc = new Object();
        this.zzd = true;
        throw new IllegalStateException("Default constructor called");
    }

    /* JADX INFO: Access modifiers changed from: private */
    public static boolean zzb(zzf zzfVar) {
        boolean z;
        if (zzfVar.zza == 2 || zzfVar.zzb != 2) {
            z = true;
        } else {
            Log.e("FaceDetector", "Contour is not supported for non-SELFIE mode.");
            z = false;
        }
        if (zzfVar.zzb == 2 && zzfVar.zzc == 1) {
            Log.e("FaceDetector", "Classification is not supported with contour.");
            return false;
        }
        return z;
    }

    @Override // com.google.android.gms.vision.Detector
    @RecentlyNonNull
    public final SparseArray<Face> detect(@RecentlyNonNull Frame frame) {
        ByteBuffer grayscaleImageData;
        Face[] zza;
        if (frame != null) {
            if (frame.getPlanes() != null && ((Image.Plane[]) Preconditions.checkNotNull(frame.getPlanes())).length == 3) {
                synchronized (this.zzc) {
                    if (this.zzd) {
                        zza = this.zzb.zza((Image.Plane[]) Preconditions.checkNotNull(frame.getPlanes()), zzs.zza(frame));
                    } else {
                        throw new IllegalStateException("Cannot use detector after release()");
                    }
                }
            } else {
                if (frame.getBitmap() != null) {
                    grayscaleImageData = zzw.zza((Bitmap) Preconditions.checkNotNull(frame.getBitmap()), true);
                } else {
                    grayscaleImageData = frame.getGrayscaleImageData();
                }
                synchronized (this.zzc) {
                    if (this.zzd) {
                        zza = this.zzb.zza((ByteBuffer) Preconditions.checkNotNull(grayscaleImageData), zzs.zza(frame));
                    } else {
                        throw new IllegalStateException("Cannot use detector after release()");
                    }
                }
            }
            HashSet hashSet = new HashSet();
            SparseArray<Face> sparseArray = new SparseArray<>(zza.length);
            int i = 0;
            for (Face face : zza) {
                int id = face.getId();
                i = Math.max(i, id);
                if (hashSet.contains(Integer.valueOf(id))) {
                    id = i + 1;
                    i = id;
                }
                hashSet.add(Integer.valueOf(id));
                sparseArray.append(this.zza.zza(id), face);
            }
            return sparseArray;
        }
        throw new IllegalArgumentException("No frame supplied.");
    }

    public final void finalize() {
        try {
            synchronized (this.zzc) {
                if (this.zzd) {
                    Log.w("FaceDetector", "FaceDetector was not released with FaceDetector.release()");
                    release();
                }
            }
        } finally {
            super.finalize();
        }
    }

    @Override // com.google.android.gms.vision.Detector
    public final boolean isOperational() {
        return this.zzb.zzb();
    }

    @Override // com.google.android.gms.vision.Detector
    public final void release() {
        super.release();
        synchronized (this.zzc) {
            if (this.zzd) {
                this.zzb.zzc();
                this.zzd = false;
            }
        }
    }

    @Override // com.google.android.gms.vision.Detector
    public final boolean setFocus(int i) {
        boolean zza;
        int zzb = this.zza.zzb(i);
        synchronized (this.zzc) {
            if (this.zzd) {
                zza = this.zzb.zza(zzb);
            } else {
                throw new RuntimeException("Cannot use detector after release()");
            }
        }
        return zza;
    }

    private FaceDetector(zzb zzbVar) {
        this.zza = new zzc();
        this.zzc = new Object();
        this.zzd = true;
        this.zzb = zzbVar;
    }
}