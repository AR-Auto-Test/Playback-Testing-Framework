package com.google.android.gms.common.server.response;

import com.google.android.gms.common.server.response.FastParser;
import java.io.BufferedReader;
import java.math.BigInteger;

/* compiled from: com.google.android.gms:play-services-base@@17.4.0 */
/* loaded from: classes.dex */
public final class zaf implements FastParser.zaa<BigInteger> {
    /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
    @Override // com.google.android.gms.common.server.response.FastParser.zaa
    public final /* synthetic */ BigInteger zaa(FastParser fastParser, BufferedReader bufferedReader) {
        BigInteger zaf;
        zaf = fastParser.zaf(bufferedReader);
        return zaf;
    }
}