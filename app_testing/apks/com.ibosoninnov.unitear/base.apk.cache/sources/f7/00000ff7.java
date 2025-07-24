package com.google.android.gms.internal.clearcut;

import com.google.android.gms.internal.clearcut.zzas;
import com.google.android.gms.internal.clearcut.zzat;

/* loaded from: classes.dex */
public abstract class zzat<MessageType extends zzas<MessageType, BuilderType>, BuilderType extends zzat<MessageType, BuilderType>> implements zzdp {
    public abstract BuilderType zza(MessageType messagetype);

    /* JADX DEBUG: Multi-variable search result rejected for r1v0, resolved type: com.google.android.gms.internal.clearcut.zzat<MessageType extends com.google.android.gms.internal.clearcut.zzas<MessageType, BuilderType>, BuilderType extends com.google.android.gms.internal.clearcut.zzat<MessageType, BuilderType>> */
    /* JADX WARN: Multi-variable type inference failed */
    @Override // com.google.android.gms.internal.clearcut.zzdp
    public final /* synthetic */ zzdp zza(zzdo zzdoVar) {
        if (zzbe().getClass().isInstance(zzdoVar)) {
            return zza((zzat<MessageType, BuilderType>) ((zzas) zzdoVar));
        }
        throw new IllegalArgumentException("mergeFrom(MessageLite) can only merge messages of the same type.");
    }

    /* JADX DEBUG: Method merged with bridge method */
    @Override // 
    /* renamed from: zzt */
    public abstract BuilderType clone();
}