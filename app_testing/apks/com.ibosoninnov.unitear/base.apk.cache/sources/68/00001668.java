package com.google.android.gms.vision.face.internal.client;

import android.os.Parcel;
import android.os.Parcelable;
import com.google.android.gms.common.internal.safeparcel.SafeParcelReader;

/* compiled from: com.google.android.gms:play-services-vision@@20.1.3 */
/* loaded from: classes.dex */
public final class zzd implements Parcelable.Creator<FaceParcel> {
    /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
    @Override // android.os.Parcelable.Creator
    public final /* synthetic */ FaceParcel createFromParcel(Parcel parcel) {
        int validateObjectHeader = SafeParcelReader.validateObjectHeader(parcel);
        LandmarkParcel[] landmarkParcelArr = null;
        zza[] zzaVarArr = null;
        int i = 0;
        int i2 = 0;
        float f2 = Float.MAX_VALUE;
        float f3 = Float.MAX_VALUE;
        float f4 = Float.MAX_VALUE;
        float f5 = 0.0f;
        float f6 = 0.0f;
        float f7 = 0.0f;
        float f8 = 0.0f;
        float f9 = 0.0f;
        float f10 = 0.0f;
        float f11 = 0.0f;
        float f12 = -1.0f;
        while (parcel.dataPosition() < validateObjectHeader) {
            int readHeader = SafeParcelReader.readHeader(parcel);
            switch (SafeParcelReader.getFieldId(readHeader)) {
                case 1:
                    i = SafeParcelReader.readInt(parcel, readHeader);
                    break;
                case 2:
                    i2 = SafeParcelReader.readInt(parcel, readHeader);
                    break;
                case 3:
                    f5 = SafeParcelReader.readFloat(parcel, readHeader);
                    break;
                case 4:
                    f6 = SafeParcelReader.readFloat(parcel, readHeader);
                    break;
                case 5:
                    f7 = SafeParcelReader.readFloat(parcel, readHeader);
                    break;
                case 6:
                    f8 = SafeParcelReader.readFloat(parcel, readHeader);
                    break;
                case 7:
                    f2 = SafeParcelReader.readFloat(parcel, readHeader);
                    break;
                case 8:
                    f3 = SafeParcelReader.readFloat(parcel, readHeader);
                    break;
                case 9:
                    landmarkParcelArr = (LandmarkParcel[]) SafeParcelReader.createTypedArray(parcel, readHeader, LandmarkParcel.CREATOR);
                    break;
                case 10:
                    f9 = SafeParcelReader.readFloat(parcel, readHeader);
                    break;
                case 11:
                    f10 = SafeParcelReader.readFloat(parcel, readHeader);
                    break;
                case 12:
                    f11 = SafeParcelReader.readFloat(parcel, readHeader);
                    break;
                case 13:
                    zzaVarArr = (zza[]) SafeParcelReader.createTypedArray(parcel, readHeader, zza.CREATOR);
                    break;
                case 14:
                    f4 = SafeParcelReader.readFloat(parcel, readHeader);
                    break;
                case 15:
                    f12 = SafeParcelReader.readFloat(parcel, readHeader);
                    break;
                default:
                    SafeParcelReader.skipUnknownField(parcel, readHeader);
                    break;
            }
        }
        SafeParcelReader.ensureAtEnd(parcel, validateObjectHeader);
        return new FaceParcel(i, i2, f5, f6, f7, f8, f2, f3, f4, landmarkParcelArr, f9, f10, f11, zzaVarArr, f12);
    }

    /* JADX DEBUG: Return type fixed from 'java.lang.Object[]' to match base method */
    @Override // android.os.Parcelable.Creator
    public final /* synthetic */ FaceParcel[] newArray(int i) {
        return new FaceParcel[i];
    }
}