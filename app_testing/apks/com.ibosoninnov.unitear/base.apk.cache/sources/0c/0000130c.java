package com.google.android.gms.internal.vision;

import android.os.Parcel;
import android.os.Parcelable;
import com.google.android.gms.common.internal.safeparcel.AbstractSafeParcelable;
import com.google.android.gms.common.internal.safeparcel.SafeParcelWriter;
import com.google.android.gms.common.internal.safeparcel.SafeParcelable;

/* compiled from: com.google.android.gms:play-services-vision@@20.1.3 */
@SafeParcelable.Class(creator = "WordBoxParcelCreator")
@SafeParcelable.Reserved({1})
/* loaded from: classes.dex */
public final class zzao extends AbstractSafeParcelable {
    public static final Parcelable.Creator<zzao> CREATOR = new zzar();
    @SafeParcelable.Field(id = 3)
    public final zzab zza;
    @SafeParcelable.Field(id = 5)
    public final String zzb;
    @SafeParcelable.Field(id = 7)
    public final String zzc;
    @SafeParcelable.Field(id = 2)
    private final zzal[] zzd;
    @SafeParcelable.Field(id = 4)
    private final zzab zze;
    @SafeParcelable.Field(id = 6)
    private final float zzf;
    @SafeParcelable.Field(id = 8)
    private final boolean zzg;

    @SafeParcelable.Constructor
    public zzao(@SafeParcelable.Param(id = 2) zzal[] zzalVarArr, @SafeParcelable.Param(id = 3) zzab zzabVar, @SafeParcelable.Param(id = 4) zzab zzabVar2, @SafeParcelable.Param(id = 5) String str, @SafeParcelable.Param(id = 6) float f2, @SafeParcelable.Param(id = 7) String str2, @SafeParcelable.Param(id = 8) boolean z) {
        this.zzd = zzalVarArr;
        this.zza = zzabVar;
        this.zze = zzabVar2;
        this.zzb = str;
        this.zzf = f2;
        this.zzc = str2;
        this.zzg = z;
    }

    @Override // android.os.Parcelable
    public final void writeToParcel(Parcel parcel, int i) {
        int beginObjectHeader = SafeParcelWriter.beginObjectHeader(parcel);
        SafeParcelWriter.writeTypedArray(parcel, 2, this.zzd, i, false);
        SafeParcelWriter.writeParcelable(parcel, 3, this.zza, i, false);
        SafeParcelWriter.writeParcelable(parcel, 4, this.zze, i, false);
        SafeParcelWriter.writeString(parcel, 5, this.zzb, false);
        SafeParcelWriter.writeFloat(parcel, 6, this.zzf);
        SafeParcelWriter.writeString(parcel, 7, this.zzc, false);
        SafeParcelWriter.writeBoolean(parcel, 8, this.zzg);
        SafeParcelWriter.finishObjectHeader(parcel, beginObjectHeader);
    }
}