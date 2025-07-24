package com.google.android.gms.common.server.response;

import com.google.android.gms.common.server.response.FastParser;
import java.io.BufferedReader;

/* compiled from: com.google.android.gms:play-services-base@@17.4.0 */
/* loaded from: classes.dex */
public final class zab implements FastParser.zaa<Float> {
    /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
    @Override // com.google.android.gms.common.server.response.FastParser.zaa
    public final /* synthetic */ Float zaa(FastParser fastParser, BufferedReader bufferedReader) {
        float zag;
        zag = fastParser.zag(bufferedReader);
        return Float.valueOf(zag);
    }
}