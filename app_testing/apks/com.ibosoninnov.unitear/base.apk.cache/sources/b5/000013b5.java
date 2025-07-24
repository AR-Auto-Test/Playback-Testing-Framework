package com.google.android.gms.internal.vision;

import com.google.android.gms.internal.vision.zzhe;
import com.google.android.gms.internal.vision.zzhf;
import java.io.IOException;

/* compiled from: com.google.android.gms:play-services-vision-common@@19.1.3 */
/* loaded from: classes.dex */
public abstract class zzhe<MessageType extends zzhf<MessageType, BuilderType>, BuilderType extends zzhe<MessageType, BuilderType>> implements zzkn {
    /* JADX DEBUG: Method merged with bridge method */
    @Override // 
    /* renamed from: zza */
    public abstract BuilderType clone();

    public abstract BuilderType zza(MessageType messagetype);

    public abstract BuilderType zza(zzif zzifVar, zzio zzioVar);

    public BuilderType zza(byte[] bArr, int i, int i2, zzio zzioVar) {
        try {
            zzif zza = zzif.zza(bArr, 0, i2, false);
            zza(zza, zzioVar);
            zza.zza(0);
            return this;
        } catch (zzjk e2) {
            throw e2;
        } catch (IOException e3) {
            String name = getClass().getName();
            StringBuilder sb = new StringBuilder("byte array".length() + name.length() + 60);
            sb.append("Reading ");
            sb.append(name);
            sb.append(" from a ");
            sb.append("byte array");
            sb.append(" threw an IOException (should never happen).");
            throw new RuntimeException(sb.toString(), e3);
        }
    }

    /* JADX DEBUG: Multi-variable search result rejected for r1v0, resolved type: com.google.android.gms.internal.vision.zzhe<MessageType extends com.google.android.gms.internal.vision.zzhf<MessageType, BuilderType>, BuilderType extends com.google.android.gms.internal.vision.zzhe<MessageType, BuilderType>> */
    /* JADX WARN: Multi-variable type inference failed */
    @Override // com.google.android.gms.internal.vision.zzkn
    public final /* synthetic */ zzkn zza(zzkk zzkkVar) {
        if (zzr().getClass().isInstance(zzkkVar)) {
            return zza((zzhe<MessageType, BuilderType>) ((zzhf) zzkkVar));
        }
        throw new IllegalArgumentException("mergeFrom(MessageLite) can only merge messages of the same type.");
    }
}