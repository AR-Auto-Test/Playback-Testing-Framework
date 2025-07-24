package com.google.android.gms.phenotype;

import android.os.Parcel;
import android.os.Parcelable;
import c.b.a.a.a;
import com.google.android.gms.common.annotation.KeepForSdk;
import com.google.android.gms.common.internal.safeparcel.AbstractSafeParcelable;
import com.google.android.gms.common.internal.safeparcel.SafeParcelWriter;
import com.google.android.gms.common.internal.safeparcel.SafeParcelable;
import java.util.Arrays;
import java.util.Map;
import java.util.TreeMap;

@KeepForSdk
@SafeParcelable.Class(creator = "ConfigurationCreator")
@SafeParcelable.Reserved({1})
/* loaded from: classes.dex */
public class Configuration extends AbstractSafeParcelable implements Comparable<Configuration> {
    @KeepForSdk
    public static final Parcelable.Creator<Configuration> CREATOR = new zzc();
    @SafeParcelable.Field(id = 2)
    private final int zzc;
    @SafeParcelable.Field(id = 3)
    private final zzi[] zzd;
    @SafeParcelable.Field(id = 4)
    private final String[] zze;
    private final Map<String, zzi> zzf = new TreeMap();

    @SafeParcelable.Constructor
    public Configuration(@SafeParcelable.Param(id = 2) int i, @SafeParcelable.Param(id = 3) zzi[] zziVarArr, @SafeParcelable.Param(id = 4) String[] strArr) {
        this.zzc = i;
        this.zzd = zziVarArr;
        for (zzi zziVar : zziVarArr) {
            this.zzf.put(zziVar.name, zziVar);
        }
        this.zze = strArr;
        if (strArr != null) {
            Arrays.sort(strArr);
        }
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
    @Override // java.lang.Comparable
    public /* synthetic */ int compareTo(Configuration configuration) {
        return this.zzc - configuration.zzc;
    }

    public boolean equals(Object obj) {
        if (obj instanceof Configuration) {
            Configuration configuration = (Configuration) obj;
            if (this.zzc == configuration.zzc && zzn.equals(this.zzf, configuration.zzf) && Arrays.equals(this.zze, configuration.zze)) {
                return true;
            }
        }
        return false;
    }

    public String toString() {
        StringBuilder sb = new StringBuilder("Configuration(");
        sb.append(this.zzc);
        sb.append(", ");
        sb.append("(");
        for (zzi zziVar : this.zzf.values()) {
            sb.append(zziVar);
            sb.append(", ");
        }
        sb.append(")");
        sb.append(", ");
        sb.append("(");
        String[] strArr = this.zze;
        if (strArr != null) {
            for (String str : strArr) {
                sb.append(str);
                sb.append(", ");
            }
        } else {
            sb.append("null");
        }
        return a.v(sb, ")", ")");
    }

    @Override // android.os.Parcelable
    public void writeToParcel(Parcel parcel, int i) {
        int beginObjectHeader = SafeParcelWriter.beginObjectHeader(parcel);
        SafeParcelWriter.writeInt(parcel, 2, this.zzc);
        SafeParcelWriter.writeTypedArray(parcel, 3, this.zzd, i, false);
        SafeParcelWriter.writeStringArray(parcel, 4, this.zze, false);
        SafeParcelWriter.finishObjectHeader(parcel, beginObjectHeader);
    }
}