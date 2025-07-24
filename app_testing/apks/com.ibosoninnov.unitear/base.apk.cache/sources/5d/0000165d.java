package com.google.android.gms.vision.face;

import android.graphics.PointF;
import androidx.annotation.RecentlyNonNull;
import com.google.android.gms.common.annotation.KeepForSdk;
import com.google.android.gms.common.internal.ShowFirstParty;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.util.Arrays;
import java.util.List;

/* compiled from: com.google.android.gms:play-services-vision@@20.1.3 */
/* loaded from: classes.dex */
public class Face {
    public static final float UNCOMPUTED_PROBABILITY = -1.0f;
    private int zza;
    private PointF zzb;
    private float zzc;
    private float zzd;
    private float zze;
    private float zzf;
    private float zzg;
    private List<Landmark> zzh;
    private final List<Contour> zzi;
    private float zzj;
    private float zzk;
    private float zzl;
    private final float zzm;

    public Face(int i, @RecentlyNonNull PointF pointF, float f2, float f3, float f4, float f5, float f6, @RecentlyNonNull Landmark[] landmarkArr, @RecentlyNonNull Contour[] contourArr, float f7, float f8, float f9, float f10) {
        this.zza = i;
        this.zzb = pointF;
        this.zzc = f2;
        this.zzd = f3;
        this.zze = f4;
        this.zzf = f5;
        this.zzg = f6;
        this.zzh = Arrays.asList(landmarkArr);
        this.zzi = Arrays.asList(contourArr);
        this.zzj = zza(f7);
        this.zzk = zza(f8);
        this.zzl = zza(f9);
        this.zzm = zza(f10);
    }

    private static float zza(float f2) {
        if (f2 < StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD || f2 > 1.0f) {
            return -1.0f;
        }
        return f2;
    }

    @RecentlyNonNull
    public List<Contour> getContours() {
        return this.zzi;
    }

    @ShowFirstParty
    @KeepForSdk
    public float getEulerX() {
        return this.zzg;
    }

    public float getEulerY() {
        return this.zze;
    }

    public float getEulerZ() {
        return this.zzf;
    }

    public float getHeight() {
        return this.zzd;
    }

    public int getId() {
        return this.zza;
    }

    public float getIsLeftEyeOpenProbability() {
        return this.zzj;
    }

    public float getIsRightEyeOpenProbability() {
        return this.zzk;
    }

    public float getIsSmilingProbability() {
        return this.zzl;
    }

    @RecentlyNonNull
    public List<Landmark> getLandmarks() {
        return this.zzh;
    }

    @RecentlyNonNull
    public PointF getPosition() {
        PointF pointF = this.zzb;
        return new PointF(pointF.x - (this.zzc / 2.0f), pointF.y - (this.zzd / 2.0f));
    }

    public float getWidth() {
        return this.zzc;
    }
}